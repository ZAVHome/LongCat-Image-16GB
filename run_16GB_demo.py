import torch
import requests
from PIL import Image
from transformers import AutoModel, AutoProcessor
from longcat_image.models.longcat_image_dit import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImageEditPipeline
import os
import gc
import traceback

# ==============================================================================
# Configuration
# ==============================================================================
# Use bfloat16 as it is the native precision for the Flux model and modern GPUs.
dtype = torch.bfloat16
checkpoint_dir = './weights/LongCat-Image-Edit'
input_image_path = "input.jpg"
output_image_path = "result_16gb_demo.png"

print(f"🚀 Launching on device: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# 1. Load Components into RAM (CPU)
# ==============================================================================
# Strategy: 
# To fit into 16GB VRAM, we avoid loading the entire model onto the GPU at once.
# Instead, we load components into System RAM (CPU) first.
# The `pipeline_longcat_image_edit.py` has been patched to manually manage
# moving the Transformer and Text Encoder to/from the GPU during the generation loop.

print("📦 Loading Qwen (Text Encoder) into RAM (this requires ~15 GB System RAM)...")

# low_cpu_mem_usage=False -> Load directly into RAM (avoiding meta device issues)
# device_map=None         -> Do NOT automatically dispatch to GPU yet
text_encoder = AutoModel.from_pretrained(
    f"{checkpoint_dir}/text_encoder", 
    torch_dtype=dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=False,
    device_map=None 
)

try:
    text_processor = AutoProcessor.from_pretrained(f"{checkpoint_dir}/text_encoder", trust_remote_code=True)
except Exception:
    print("⚠️ Local text processor not found, falling back to HuggingFace hub...")
    text_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

# ==============================================================================
# 2. Load Transformer into RAM (CPU)
# ==============================================================================
print("🤖 Loading Transformer into RAM...")

transformer = LongCatImageTransformer2DModel.from_pretrained(
    f"{checkpoint_dir}/transformer",
    torch_dtype=dtype,
    low_cpu_mem_usage=False
)

# ==============================================================================
# 3. Assemble Pipeline and Enable Offloading
# ==============================================================================
print("🔧 Assembling pipeline...")

pipe = LongCatImageEditPipeline.from_pretrained(
    checkpoint_dir,
    text_encoder=text_encoder,
    text_processor=text_processor,
    transformer=transformer,
    torch_dtype=dtype
)

print("🧠 Enabling Model CPU Offload...")
# Crucial Step:
# enable_model_cpu_offload() registers hooks that move submodules to the GPU
# only when their 'forward' method is called, and back to CPU afterwards.
# HOWEVER, our custom pipeline adds specific "force offload" logic for the 
# heavy Text Encoder and Transformer to ensure they don't overlap in VRAM.
pipe.enable_model_cpu_offload()

# Enable VAE slicing/tiling to save memory during the final image decoding phase
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# ==============================================================================
# 4. Image Generation
# ==============================================================================
# Load or download input image
if not os.path.exists(input_image_path):
    print("📥 Input image not found, downloading sample...")
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        image.save(input_image_path)
    except Exception as e:
        print(f"❌ Failed to download sample image: {e}")
        exit(1)
else:
    image = Image.open(input_image_path).convert("RGB")

print("✨ Starting generation...")
prompt = "make it look like a snowy winter day, high quality, 4k"
generator = torch.Generator(device="cpu").manual_seed(42)

try:
    # Explicit garbage collection before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    output = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=30, 
        guidance_scale=7.0,
        generator=generator
    ).images[0]

    output.save(output_image_path)
    print(f"🎉 SUCCESS! Result saved to: {output_image_path}")

except Exception as e:
    print(f"\n❌ ERROR during generation: {e}")
    traceback.print_exc()

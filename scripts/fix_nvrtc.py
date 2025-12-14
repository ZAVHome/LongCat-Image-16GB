import os
import sys
import glob
import shutil
from pathlib import Path

def find_nvrtc_lib(env_prefix):
    """Search for libnvrtc-builtins.so in site-packages/nvidia."""
    # Common locations in site-packages
    site_packages = glob.glob(os.path.join(env_prefix, "lib", "python*", "site-packages"))
    if not site_packages:
        print("❌ Could not locate site-packages directory.")
        return None
    
    site_packages = site_packages[0]
    
    # Search recursively for the library
    search_pattern = os.path.join(site_packages, "nvidia", "**", "libnvrtc-builtins.so*")
    candidates = glob.glob(search_pattern, recursive=True)
    
    if not candidates:
        return None
        
    # Sort by version number extracted from filename (e.g., .13.0 > .12.4)
    def version_key(path):
        try:
            # Extract distinct version part (e.g. from .so.13.0 -> 13.0)
            # Find the last part that looks like a version
            parts = path.split(".so.")
            if len(parts) > 1:
                ver_str = parts[-1]
                return [int(p) for p in ver_str.split('.') if p.isdigit()]
        except:
            pass
        return [0]

    candidates.sort(key=version_key, reverse=True)
    return candidates[0]

def main():
    print("🔧 LongCat-Image Environment Fixer")
    print("===================================")
    
    # 1. Identify Environment
    env_prefix = sys.prefix
    print(f"📍 Current Environment: {env_prefix}")
    print(f"🐍 Python Executable: {sys.executable}")
    
    # 2. Check for missing library
    target_lib_dir = os.path.join(env_prefix, "lib")
    target_link_name = os.path.join(target_lib_dir, "libnvrtc-builtins.so.13.0") # Target name often expected by NVRTC
    
    # Also check for unversioned or other versioned names if needed, but the error specifically asked for .13.0
    # or .12.x depending on the CUDA version. We will create a symlink with the exact name of the source file first.
    
    source_lib = find_nvrtc_lib(env_prefix)
    
    if not source_lib:
        print("❌ Could not find 'libnvrtc-builtins.so' in site-packages.")
        print("   Please ensure you have installed the nvidia-cuda-nvrtc-cuXX package.")
        sys.exit(1)
        
    print(f"🔎 Found source library: {source_lib}")
    
    source_name = os.path.basename(source_lib)
    destination = os.path.join(target_lib_dir, source_name)
    
    # 3. Create Symlink
    if os.path.exists(destination):
        if os.path.islink(destination):
            current_target = os.readlink(destination)
            if current_target == source_lib:
                print("✅ Fix already applied (symlink exists and is correct).")
                sys.exit(0)
            else:
                print(f"⚠️  Symlink exists but points to {current_target}. Updating...")
                os.remove(destination)
        else:
             print(f"⚠️  File {destination} exists and is not a symlink. Skipping to avoid data loss.")
             sys.exit(1)

    try:
        os.symlink(source_lib, destination)
        print(f"✅ Created symlink: {destination} -> {source_lib}")
        
        # 4. Handle specific version requirement if needed (e.g. error asked for .13.0 explicitly)
        # If the found file is .13.0, we are good. If it's something else, we might need an alias.
        # But usually NVRTC loads based on the major version.
        
    except OSError as e:
        print(f"❌ Failed to create symlink: {e}")
        sys.exit(1)

    print("\n🎉 Environment fixed! You should now be able to run the demo.")

if __name__ == "__main__":
    main()

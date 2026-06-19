#!/usr/bin/env python3
"""
Upload st-evidence-instruct dataset to Hugging Face with dereferenced symlinks.
"""

import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, login

def copy_with_dereferenced_links(src_dir, dst_dir):
    """
    Copy directory structure, dereferencing symbolic links.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    print(f"Copying {src_dir} to {dst_dir}...")
    print("This may take a while for large datasets...")

    # Walk through source directory
    for item in src_path.rglob('*'):
        relative_path = item.relative_to(src_path)
        dest_item = dst_path / relative_path

        if item.is_file():
            # Create parent directory if needed
            dest_item.parent.mkdir(parents=True, exist_ok=True)

            # Copy file (this automatically dereferences symlinks)
            print(f"  Copying: {relative_path}")
            shutil.copy2(item, dest_item, follow_symlinks=True)
        elif item.is_dir() and not item.is_symlink():
            # Create directory
            dest_item.mkdir(parents=True, exist_ok=True)
        elif item.is_symlink() and item.is_dir():
            # Handle directory symlinks - copy the target directory contents
            print(f"  Dereferencing symlink: {relative_path} -> {os.readlink(item)}")
            dest_item.mkdir(parents=True, exist_ok=True)
            # Recursively copy the symlink target
            shutil.copytree(item, dest_item, dirs_exist_ok=True, symlinks=False)

def main():
    # Configuration
    source_dir = "/fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct"
    repo_id = "shijiewang/st-evidence-instruct"  # Change this to your HF username

    print("=" * 60)
    print("ST-Evidence-Instruct Dataset Upload to Hugging Face")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Target repo: {repo_id}")
    print()

    # Login to Hugging Face
    print("Step 1: Logging in to Hugging Face...")
    try:
        login()
        print("✓ Login successful")
    except Exception as e:
        print(f"✗ Login failed: {e}")
        print("Please run 'huggingface-cli login' first")
        return

    # Create temporary directory
    print("\nStep 2: Creating temporary directory for upload...")
    temp_dir = tempfile.mkdtemp(prefix="st-evidence-instruct-")
    print(f"✓ Created: {temp_dir}")

    try:
        # Copy data with dereferenced symlinks
        print("\nStep 3: Copying data (dereferencing symlinks)...")
        copy_with_dereferenced_links(source_dir, temp_dir)
        print("✓ Data copy complete")

        # Show what we're uploading
        print("\nStep 4: Checking directory structure...")
        os.system(f"du -sh {temp_dir}/*")

        # Create/upload to Hugging Face
        print(f"\nStep 5: Uploading to Hugging Face ({repo_id})...")
        api = HfApi()

        # Create repository
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            print(f"✓ Repository created/verified: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"✗ Failed to create repository: {e}")
            return

        # Upload folder
        print("Uploading files... (this may take a while for large datasets)")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="dataset",
        )

        print("\n" + "=" * 60)
        print("✓ Upload complete!")
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during upload: {e}")
        raise
    finally:
        # Cleanup
        print("\nStep 6: Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()

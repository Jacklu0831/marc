# python burst/copy_rearc_conceptarc.py /scratch/yl11330/re-arc /scratch/yl11330/marc/re-arc
# python burst/copy_rearc_conceptarc.py /scratch/yl11330/ConceptARC /scratch/yl11330/marc/ConceptARC

#!/usr/bin/env python3
import os
import sys
import subprocess

def copy_with_cat(source, destination):
    # Ensure the source directory exists
    if not os.path.isdir(source):
        print(f"Error: Source directory '{source}' does not exist or is not a directory.")
        sys.exit(1)

    # Walk through the source directory recursively
    for root, dirs, files in os.walk(source):
        for file in files:
            src_file = os.path.join(root, file)
            # Determine the relative path and construct the corresponding destination path
            rel_path = os.path.relpath(src_file, source)
            dst_file = os.path.join(destination, rel_path)
            dst_dir = os.path.dirname(dst_file)

            # Create destination directory if it doesn't exist
            os.makedirs(dst_dir, exist_ok=True)

            # Build and run the cat command to copy the file's content
            # Note: we wrap paths in single quotes in case they contain spaces
            cmd = f"cat '{src_file}' > '{dst_file}'"
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"Error copying {src_file} to {dst_file}")
                sys.exit(result.returncode)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} <source_directory> <destination_directory>".format(sys.argv[0]))
        sys.exit(1)

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    copy_with_cat(src_dir, dst_dir)

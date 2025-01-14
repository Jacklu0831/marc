import glob
import os
import tarfile
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='train_outputs')
parser.add_argument('--date_start', type=int, default=1101)
parser.add_argument('--date_end', type=int, required=True)

args = parser.parse_args()


def tar_and_remove(d):
    if os.path.isdir(d):
        tar_path = f"{d}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(d, arcname=os.path.basename(d))  # Add directory to tar.gz
        shutil.rmtree(d)  # Remove the original directory
        print(f"Compressed and removed: {d}")
    else:
        print(f"Skipping: {d} (not a directory)")

# Example usage:
for date in range(args.date_start, args.date_end + 1):
    for d in glob.glob(f'{args.root_dir}/{str(date)}*'):
        tar_and_remove(d)
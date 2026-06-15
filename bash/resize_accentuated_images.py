"""
Resize all accentuated PNG images (1024x1024) to 425x425 and save to a
staging directory, preserving the original folder structure.

Run on HPC (CPU-only, no GPU needed):
    python resize_accentuated_images.py

Or submit via slurm:
    sbatch resize_accentuated_images.sh
"""

import os
import glob
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

SRC_ROOT = "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs"
DST_ROOT = "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs_425px"
TARGET_SIZE = (425, 425)
N_WORKERS = 16

# Only process *_accentuation dirs (skip gifs)
DIR_PATTERN = "*_accentuation"

# ── Worker ────────────────────────────────────────────────────────────────────

def resize_one(src_path, dst_path):
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if os.path.exists(dst_path):
            return "skip"
        img = Image.open(src_path).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(dst_path, format="PNG", optimize=True)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def collect_jobs():
    jobs = []
    for acc_dir in sorted(glob.glob(os.path.join(SRC_ROOT, DIR_PATTERN))):
        dir_name = os.path.basename(acc_dir)
        dst_dir = os.path.join(DST_ROOT, dir_name)
        for src_path in glob.glob(os.path.join(acc_dir, "*.png")):
            fname = os.path.basename(src_path)
            dst_path = os.path.join(dst_dir, fname)
            jobs.append((src_path, dst_path))
    return jobs


if __name__ == "__main__":
    jobs = collect_jobs()
    print(f"Found {len(jobs)} PNG files across {len(glob.glob(os.path.join(SRC_ROOT, DIR_PATTERN)))} dirs")
    print(f"Resizing to {TARGET_SIZE} -> {DST_ROOT}")

    ok = skip = errors = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {exe.submit(resize_one, src, dst): (src, dst) for src, dst in jobs}
        for fut in tqdm(as_completed(futures), total=len(jobs), desc="Resizing"):
            result = fut.result()
            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                errors += 1
                print(f"  {result} — {futures[fut][0]}")

    print(f"\nDone: {ok} resized, {skip} skipped (already exist), {errors} errors")
    print(f"Output: {DST_ROOT}")

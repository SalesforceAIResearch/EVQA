#!/usr/bin/env python3
"""
Check correspondence between CSV and videos_6fps directory
Compare with videos_6fps_subset
"""

import pandas as pd
from pathlib import Path
import os

# Paths
csv_path = "/fsx/home/shijie.wang/code/EVQA/benchmark/st_evidence_gen/st_evidence_gen.csv"
videos_6fps = "/fsx/home/shijie.wang/code/EVQA/benchmark/st_evidence_gen/videos_6fps"
videos_6fps_subset = "/fsx/home/shijie.wang/code/EVQA/benchmark/st_evidence_gen/videos_6fps_subset"

print("=" * 70)
print("Comparing videos_6fps vs videos_6fps_subset vs CSV")
print("=" * 70)

# Read CSV
print("\n1. Reading CSV file...")
df = pd.read_csv(csv_path)
csv_video_paths = set(df['video_path'].unique())
print(f"   Unique video paths in CSV: {len(csv_video_paths)}")

# Scan videos_6fps
print("\n2. Scanning videos_6fps directory...")
videos_6fps_files = set()
for root, dirs, files in os.walk(videos_6fps):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            rel_path = os.path.relpath(os.path.join(root, file), videos_6fps)
            videos_6fps_files.add(rel_path)
print(f"   Video files found: {len(videos_6fps_files)}")

# Scan videos_6fps_subset
print("\n3. Scanning videos_6fps_subset directory...")
videos_6fps_subset_files = set()
for root, dirs, files in os.walk(videos_6fps_subset):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            rel_path = os.path.relpath(os.path.join(root, file), videos_6fps_subset)
            videos_6fps_subset_files.add(rel_path)
print(f"   Video files found: {len(videos_6fps_subset_files)}")

# Comparison
print("\n" + "=" * 70)
print("Comparison Results:")
print("=" * 70)

# CSV vs videos_6fps
print("\n📁 CSV vs videos_6fps:")
missing_in_6fps = csv_video_paths - videos_6fps_files
extra_in_6fps = videos_6fps_files - csv_video_paths
matched_6fps = csv_video_paths & videos_6fps_files

print(f"   Videos in CSV:                 {len(csv_video_paths)}")
print(f"   Videos in videos_6fps:         {len(videos_6fps_files)}")
print(f"   Matched:                       {len(matched_6fps)}")
print(f"   Missing from videos_6fps:      {len(missing_in_6fps)}")
print(f"   Extra in videos_6fps:          {len(extra_in_6fps)}")

if len(missing_in_6fps) == 0:
    print("   ✅ All CSV videos exist in videos_6fps")
else:
    print(f"   ❌ {len(missing_in_6fps)} CSV videos missing from videos_6fps")

# CSV vs videos_6fps_subset
print("\n📁 CSV vs videos_6fps_subset:")
missing_in_subset = csv_video_paths - videos_6fps_subset_files
extra_in_subset = videos_6fps_subset_files - csv_video_paths
matched_subset = csv_video_paths & videos_6fps_subset_files

print(f"   Videos in CSV:                 {len(csv_video_paths)}")
print(f"   Videos in videos_6fps_subset:  {len(videos_6fps_subset_files)}")
print(f"   Matched:                       {len(matched_subset)}")
print(f"   Missing from subset:           {len(missing_in_subset)}")
print(f"   Extra in subset:               {len(extra_in_subset)}")

if len(missing_in_subset) == 0:
    print("   ✅ All CSV videos exist in videos_6fps_subset")
else:
    print(f"   ❌ {len(missing_in_subset)} CSV videos missing from subset")

# videos_6fps vs videos_6fps_subset
print("\n📁 videos_6fps vs videos_6fps_subset:")
in_6fps_not_subset = videos_6fps_files - videos_6fps_subset_files
in_subset_not_6fps = videos_6fps_subset_files - videos_6fps_files
in_both = videos_6fps_files & videos_6fps_subset_files

print(f"   In videos_6fps only:           {len(in_6fps_not_subset)}")
print(f"   In videos_6fps_subset only:    {len(in_subset_not_6fps)}")
print(f"   In both:                       {len(in_both)}")

if in_subset_not_6fps:
    print(f"\n   ⚠️  WARNING: videos_6fps_subset has {len(in_subset_not_6fps)} videos NOT in videos_6fps!")
    if len(in_subset_not_6fps) <= 10:
        for v in sorted(in_subset_not_6fps):
            print(f"      - {v}")
    else:
        for v in sorted(list(in_subset_not_6fps)[:10]):
            print(f"      - {v}")
        print(f"      ... and {len(in_subset_not_6fps) - 10} more")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)

if videos_6fps_subset_files.issubset(videos_6fps_files):
    print("✅ videos_6fps_subset is a proper subset of videos_6fps")
    print(f"   (videos_6fps has {len(in_6fps_not_subset)} additional videos)")
elif videos_6fps_subset_files == videos_6fps_files:
    print("✅ videos_6fps and videos_6fps_subset are identical")
else:
    print("❌ videos_6fps_subset is NOT a subset of videos_6fps")
    print("   (subset has videos not in the full set)")

if len(missing_in_subset) == 0 and len(missing_in_6fps) == 0:
    print("\n✅ Both directories contain all CSV videos")
elif len(missing_in_subset) == 0:
    print("\n✅ videos_6fps_subset contains all CSV videos")
    print("⚠️  videos_6fps is missing some CSV videos")
elif len(missing_in_6fps) == 0:
    print("\n✅ videos_6fps contains all CSV videos")
    print("⚠️  videos_6fps_subset is missing some CSV videos")
else:
    print("\n❌ Both directories are missing some CSV videos")

print("=" * 70)

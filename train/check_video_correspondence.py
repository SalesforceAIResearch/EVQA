#!/usr/bin/env python3
"""
Check if videos in CSV match with actual video files in videos_6fps_subset directory
"""

import pandas as pd
from pathlib import Path
import os

# Paths
csv_path = "/fsx/home/shijie.wang/code/EVQA/benchmark/st_evidence_gen/st_evidence_gen.csv"
video_dir = "/fsx/home/shijie.wang/code/EVQA/benchmark/st_evidence_gen/videos_6fps_subset"

print("=" * 70)
print("Checking Video Correspondence")
print("=" * 70)

# Read CSV
print("\n1. Reading CSV file...")
df = pd.read_csv(csv_path)
print(f"   Total rows in CSV: {len(df)}")

# Get unique video paths from CSV
csv_video_paths = df['video_path'].unique()
print(f"   Unique video paths in CSV: {len(csv_video_paths)}")

# Get all video files from directory
print("\n2. Scanning video directory...")
actual_videos = set()
for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Get relative path from video_dir
            rel_path = os.path.relpath(os.path.join(root, file), video_dir)
            actual_videos.add(rel_path)

print(f"   Video files found: {len(actual_videos)}")

# Convert CSV paths to set for comparison
csv_videos = set(csv_video_paths)

# Compare
print("\n3. Comparison Results:")
print("-" * 70)

# Videos in CSV but not in directory
missing_in_dir = csv_videos - actual_videos
if missing_in_dir:
    print(f"\n❌ Videos in CSV but NOT in directory: {len(missing_in_dir)}")
    if len(missing_in_dir) <= 10:
        for v in sorted(missing_in_dir):
            print(f"   - {v}")
    else:
        for v in sorted(list(missing_in_dir)[:10]):
            print(f"   - {v}")
        print(f"   ... and {len(missing_in_dir) - 10} more")
else:
    print("\n✅ All videos in CSV exist in directory")

# Videos in directory but not in CSV
extra_in_dir = actual_videos - csv_videos
if extra_in_dir:
    print(f"\n⚠️  Videos in directory but NOT in CSV: {len(extra_in_dir)}")
    if len(extra_in_dir) <= 10:
        for v in sorted(extra_in_dir):
            print(f"   - {v}")
    else:
        for v in sorted(list(extra_in_dir)[:10]):
            print(f"   - {v}")
        print(f"   ... and {len(extra_in_dir) - 10} more")
else:
    print("\n✅ No extra videos in directory")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print(f"Videos in CSV:       {len(csv_videos)}")
print(f"Videos in directory: {len(actual_videos)}")
print(f"Matched:             {len(csv_videos & actual_videos)}")
print(f"Missing from dir:    {len(missing_in_dir)}")
print(f"Extra in dir:        {len(extra_in_dir)}")

if len(missing_in_dir) == 0 and len(extra_in_dir) == 0:
    print("\n✅ Perfect match! All videos correspond one-to-one.")
elif len(missing_in_dir) == 0:
    print("\n⚠️  All CSV videos exist, but there are extra videos in directory.")
else:
    print("\n❌ Some CSV videos are missing from directory!")

print("=" * 70)

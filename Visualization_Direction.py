# Joshua Johnson
# July 14, 2026
# Direction Visualization

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage
from PIL import Image
import glob

# Directories
STATS_FILE = '/Volumes/Untitled/flash_detection/statistics/cr2106_statistics_respike1.pkl'
SEG_DIR = '/Volumes/Untitled/flash_detection/results/segs/CR2106/'
OUTPUT_DIR = '/Volumes/Untitled/flash_detection/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CR = '2106'

with open(STATS_FILE, 'rb') as f:
    stats = pickle.load(f)

centroids = stats['centroids']
print(f"Total flashes: {stats['total_flashes']:,}")
print(f"Centroids: {len(centroids)}")

# CH Centroid
seg_files = sorted(glob.glob(os.path.join(SEG_DIR, '*.tiff')))
print(f"  Found {len(seg_files)} segmentation files")

# Use first segmentation for CH centroid
seg = np.array(Image.open(seg_files[0])) > 0

labels = measure.label(seg)
areas_ch = []
for label in np.unique(labels)[1:]:
    areas_ch.append((label, (labels == label).sum()))

if not areas_ch:
    print("  No CH found!")
    exit()

largest_label = max(areas_ch, key=lambda x: x[1])[0]
ch_mask = (labels == largest_label)
y, x = np.where(ch_mask)
ch_centroid = (x.mean(), y.mean())

print(f"  CH centroid: ({ch_centroid[0]:.1f}, {ch_centroid[1]:.1f})")

# Flash directions
# 100000 Sampled centroids
sample_size = min(100000, len(centroids))
sample_idx = np.random.choice(len(centroids), sample_size, replace=False)
sample_centroids = centroids[sample_idx]

flash_directions = []
for cx, cy in sample_centroids:
    dx = cx - ch_centroid[0]
    dy = cy - ch_centroid[1]
    angle = np.arctan2(dy, dx)
    flash_directions.append(angle)

flash_directions = np.array(flash_directions)

print(f"  Flash directions: {len(flash_directions)}")

# Boundary detection
print("\n[Computing boundary directions...]")

sample_segs = seg_files[::10]
print(f"  Sampling {len(sample_segs)} segmentations")

boundary_directions = []

for i, seg_file in enumerate(sample_segs):
    if i % 20 == 0:
        print(f"    Processing boundary {i+1}/{len(sample_segs)}...")
    
    try:
        seg = np.array(Image.open(seg_file)) > 0
        
        # Find boundary pixels
        from scipy.ndimage import binary_dilation, binary_erosion
        dilated = binary_dilation(seg)
        eroded = binary_erosion(seg)
        boundary = dilated & ~eroded
        
        # Get boundary pixel coordinates
        y_bound, x_bound = np.where(boundary)
        
        # Compute direction for each boundary pixel
        for by, bx in zip(y_bound, x_bound):
            dx = bx - ch_centroid[0]
            dy = by - ch_centroid[1]
            angle = np.arctan2(dy, dx)
            boundary_directions.append(angle)
            
    except Exception as e:
        print(f"    Error: {e}")
        continue

boundary_directions = np.array(boundary_directions)
print(f"  Boundary directions: {len(boundary_directions)}")

# Normalized direction histogram
# Bins in radians from -π to π
bins_dir = np.arange(-np.pi, np.pi + 0.1, 0.1)

flash_hist_dir, _ = np.histogram(flash_directions, bins=bins_dir)
boundary_hist_dir, _ = np.histogram(boundary_directions, bins=bins_dir)
normalized_direction = flash_hist_dir / (boundary_hist_dir + 1e-10)

print(f"  Normalized direction mean: {np.mean(normalized_direction):.3f}")
print(f"  Normalized direction max: {np.max(normalized_direction):.3f}")

# Raw direction histogram
plt.figure(figsize=(10, 6))
plt.hist(flash_directions, bins=bins_dir, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Leading Edge (0 rad)')
plt.axvline(x=np.pi, color='red', linestyle='--', linewidth=2)
plt.axvline(x=-np.pi, color='red', linestyle='--', linewidth=2, label='Trailing Edge (±π rad)')
plt.axvline(x=np.pi/2, color='green', linestyle='--', linewidth=2, label='Poleward (π/2 rad)')
plt.axvline(x=-np.pi/2, color='green', linestyle='--', linewidth=2, label='Equatorward (-π/2 rad)')
plt.xlabel('Direction Relative to CH Centroid (radians)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'CR{CR} Flash Direction Distribution\n{sample_size} sampled flashes', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure4a_direction_raw.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure4a_direction_raw.png")

# Boundary direction histogram
plt.figure(figsize=(10, 6))
plt.hist(boundary_directions, bins=bins_dir, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Leading Edge (0 rad)')
plt.axvline(x=np.pi, color='red', linestyle='--', linewidth=2)
plt.axvline(x=-np.pi, color='red', linestyle='--', linewidth=2, label='Trailing Edge (±π rad)')
plt.axvline(x=np.pi/2, color='green', linestyle='--', linewidth=2, label='Poleward (π/2 rad)')
plt.axvline(x=-np.pi/2, color='green', linestyle='--', linewidth=2, label='Equatorward (-π/2 rad)')
plt.xlabel('Direction Relative to CH Centroid (radians)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'CR{CR} Boundary Pixel Direction Distribution\n{len(boundary_directions):,} boundary pixels', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure4b_boundary_direction.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure4b_boundary_direction.png")

# Normalized histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
bar_width = 0.1
ax.bar(bins_dir[:-1], normalized_direction, width=bar_width, 
       color='blue', alpha=0.7, edgecolor='black')

ax.axhline(y=np.mean(normalized_direction), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(normalized_direction):.3f}')

ax.set_ylim(0, 0.18)

ax.set_xlabel('Direction Relative to CH Centroid (radians)', fontsize=12)
ax.set_ylabel('Normalized Count', fontsize=12)
ax.set_title(f'CR{CR} Normalized Flash Direction', fontsize=14)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=np.pi, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=-np.pi, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=np.pi/2, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
ax.axvline(x=-np.pi/2, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

ax.text(0.02, 0.92, f'Mean = {np.mean(normalized_direction):.3f}', 
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure4c_normalized_direction.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure4c_normalized_direction.png")
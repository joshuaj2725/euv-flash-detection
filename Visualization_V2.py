# Joshua Johnson
# July 7, 2026
# Visualization V2

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage
from PIL import Image
import glob

# Directories
STATS_FILE = '/Volumes/Untitled/flash_detection/statistics/cr2099_statistics_respike0.pkl'
SEG_DIR = '/Volumes/Untitled/flash_detection/results/segs/CR2099/'
OUTPUT_DIR = '/Volumes/Untitled/flash_detection/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CR = '2099'

with open(STATS_FILE, 'rb') as f:
    stats = pickle.load(f)

areas = stats['areas']
distances = stats['distances']
centroids = stats['centroids']
intensities = stats['intensities'] if 'intensities' in stats else None

print(f"Total flashes: {stats['total_flashes']:,}")
print(f"Mean area: {stats['mean_area']:.2f} pixels")
print(f"Mean distance: {stats['mean_distance']:.2f} pixels")
if intensities is not None:
    print(f"Mean intensity: {stats['mean_intensity']:.2f}")

# All-pixel histogram
seg_files = sorted(glob.glob(os.path.join(SEG_DIR, '*.tiff')))
print(f"  Found {len(seg_files)} segmentation files")

# Sample every 10th segmentation for speed
sample_segs = seg_files[::10]
print(f"  Sampling {len(sample_segs)} segmentations")

all_pixel_distances = []
bins = np.arange(-20, 21, 1)

for i, seg_file in enumerate(sample_segs):
    if i % 20 == 0:
        print(f"    Processing {i+1}/{len(sample_segs)}...")
    
    try:
        seg = np.array(Image.open(seg_file)) > 0
        
        # Compute signed distance
        dist = (ndimage.distance_transform_edt(~seg) - 
                ndimage.distance_transform_edt(seg) + seg - 0.5)
        
        dist_flat = dist.flatten()
        dist_flat = dist_flat[(dist_flat >= -20) & (dist_flat <= 20)]
        all_pixel_distances.extend(dist_flat)
        
    except Exception as e:
        print(f"    Error: {e}")
        continue

all_pixel_distances = np.array(all_pixel_distances)
print(f"  Total pixels sampled: {len(all_pixel_distances):,}")

# Compute normalized histogram
flash_hist, _ = np.histogram(distances, bins=bins)
allpixel_hist, _ = np.histogram(all_pixel_distances, bins=bins)
normalized = flash_hist / (allpixel_hist + 1e-10)

print(f"  Normalized mean: {np.mean(normalized):.4f}")
print(f"  Normalized max: {np.max(normalized):.4f}")

# Distance Histogram
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=bins, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='CH Boundary')
plt.xlabel('Distance from CH Boundary (pixels)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'CR{CR} Flash Distance Distribution\n{stats["total_flashes"]:,} flashes', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure1_distance_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure1_distance_histogram.png")

# All-pixel histogram
plt.figure(figsize=(10, 6))
plt.hist(all_pixel_distances, bins=bins, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='CH Boundary')
plt.xlabel('Distance from CH Boundary (pixels)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'CR{CR} All-Pixel Distance Histogram\n{len(all_pixel_distances):,} pixels sampled', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure1b_allpixel_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure1b_allpixel_histogram.png")


# Normalized distance histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

bars = ax.bar(bins[:-1], normalized, width=1, color='blue', alpha=0.7, edgecolor='black')

ax.axhline(y=np.mean(normalized), color='red', linestyle='--', linewidth=2, 
           label=f'Mean = {np.mean(normalized):.4f}')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylim(0, 0.18)

ax.set_xlabel('Distance from CH Boundary (pixels)', fontsize=12)
ax.set_ylabel('Normalized Count', fontsize=12)
ax.set_title(f'CR{CR} Normalized Flash Distance', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.92, f'Mean = {np.mean(normalized):.4f}', 
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
ax.text(0.02, 0.85, 'No peak at boundary', 
        transform=ax.transAxes, fontsize=11, color='red',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure1c_normalized_distance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure1c_normalized_distance.png")

# Full scale
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

bars = ax.bar(bins[:-1], normalized, width=1, color='blue', alpha=0.7, edgecolor='black')

ax.axhline(y=1.0, color='green', linestyle=':', linewidth=2, alpha=0.7, 
           label='Uniform distribution (y=1.0)')
ax.axhline(y=np.mean(normalized), color='red', linestyle='--', linewidth=2, 
           label=f'Mean = {np.mean(normalized):.4f}')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Distance from CH Boundary (pixels)', fontsize=12)
ax.set_ylabel('Normalized Count', fontsize=12)
ax.set_title(f'CR{CR} Normalized Flash Distance (Full Scale)', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure1c_normalized_distance_full_scale.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure1c_normalized_distance_full_scale.png")

# Area histogram
plt.figure(figsize=(10, 6))
plt.hist(np.clip(areas, 1, 20), bins=range(1, 21), color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Area (pixels)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'CR{CR} Flash Area Distribution\nMean: {stats["mean_area"]:.2f} px, Median: {stats["median_area"]:.0f} px', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure2_area_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure2_area_histogram.png")

# Area vs. distance heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

heatmap, xedges, yedges = np.histogram2d(
    distances, 
    np.clip(areas, 1, 20), 
    bins=[np.arange(-20, 21, 1), np.arange(1, 21, 1)]
)

im = ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               origin='lower', cmap='jet', aspect='auto')
ax.set_xlabel('Distance from CH Boundary (pixels)', fontsize=12)
ax.set_ylabel('Area (pixels)', fontsize=12)
ax.set_title(f'CR{CR} Area vs Distance Heatmap\n{stats["total_flashes"]:,} flashes', fontsize=14)
ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure3_area_distance_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure3_area_distance_heatmap.png")

# Intensity histogram
if intensities is not None and len(intensities) > 0:
    plt.figure(figsize=(10, 6))
    plt.hist(np.clip(intensities, 0, 1000), bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Mean Flash Intensity (photon count)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'CR{CR} Flash Intensity Distribution\nMean: {stats["mean_intensity"]:.2f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure5_intensity_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cr{CR}_figure5_intensity_histogram.png")
    
    # Intensity vs Distance heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    heatmap, xedges, yedges = np.histogram2d(
        distances, 
        np.clip(intensities, 0, 1000), 
        bins=[np.arange(-20, 21, 1), np.arange(0, 1000, 20)]
    )
    
    im = ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   origin='lower', cmap='jet', aspect='auto')
    ax.set_xlabel('Distance from CH Boundary (pixels)', fontsize=12)
    ax.set_ylabel('Mean Flash Intensity (photon count)', fontsize=12)
    ax.set_title(f'CR{CR} Intensity vs Distance Heatmap\n{stats["total_flashes"]:,} flashes', fontsize=14)
    ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure6_intensity_distance_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cr{CR}_figure6_intensity_distance_heatmap.png")

# Summary
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# (a) Distance histogram
axes[0, 0].hist(distances, bins=bins, color='blue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Distance (pixels)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title(f'Distance\n{stats["total_flashes"]:,} flashes')
axes[0, 0].grid(True, alpha=0.3)

# (b) Area histogram
axes[0, 1].hist(np.clip(areas, 1, 20), bins=range(1, 21), color='blue', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Area (pixels)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title(f'Area\nMean: {stats["mean_area"]:.2f} px')
axes[0, 1].grid(True, alpha=0.3)

# (c) Normalized distance (KEY FIGURE)
axes[0, 2].bar(bins[:-1], normalized, width=1, color='blue', alpha=0.7, edgecolor='black')
axes[0, 2].axhline(y=np.mean(normalized), color='red', linestyle='--', linewidth=2)
axes[0, 2].axhline(y=1.0, color='green', linestyle=':', linewidth=1, alpha=0.5)
axes[0, 2].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 2].set_xlabel('Distance (pixels)')
axes[0, 2].set_ylabel('Normalized Count')
axes[0, 2].set_title(f'Normalized Distance\nMean: {np.mean(normalized):.4f}')
axes[0, 2].grid(True, alpha=0.3)

# (d) Area vs Distance heatmap
heatmap, _, _ = np.histogram2d(distances, np.clip(areas, 1, 20), 
                                bins=[np.arange(-20, 21, 1), np.arange(1, 21, 1)])
im = axes[1, 0].imshow(heatmap.T, extent=[-20, 20, 1, 20],
                       origin='lower', cmap='jet', aspect='auto')
axes[1, 0].set_xlabel('Distance (pixels)')
axes[1, 0].set_ylabel('Area (pixels)')
axes[1, 0].set_title('Area vs Distance')
axes[1, 0].axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
plt.colorbar(im, ax=axes[1, 0])

# (f) Pie chart
labels = ['Inside CH', 'Outside CH', 'At Boundary']
sizes = [stats['inside_ch'], stats['outside_ch'], stats['at_boundary']]
colors = ['blue', 'orange', 'green']
axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1, 2].set_title(f'Location\n{stats["total_flashes"]:,} total')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'cr{CR}_figure7_combined_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cr{CR}_figure7_combined_summary.png")

# Progress log
print("\n" + "="*60)
print(f"ALL FIGURES GENERATED FOR CR{CR}")
print(f"Output directory: {OUTPUT_DIR}")
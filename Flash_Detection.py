# Joshua Johnson
# April 20, 2026
# Flash Detection

import os
import sys
import glob
import numpy as np
import pickle
import time
import gc
from astropy.io import fits
from scipy import ndimage
from skimage import morphology, measure
from PIL import Image

CR = '2099'
RESPIKE = 0 # 0 if despiked, 1 if respiked
MIN_AREA = 1.0
MAX_DIST = 20.0 # Max distance from boundary for computational efficiency

# Optimize parameters for lower computational costs
MAX_FLASHES_PER_FRAME = None
MAX_FRAMES_TO_PROCESS = None 

# Directories
BASE_DIR = '/Volumes/Untitled/flash_detection/'
DATA_DIR = '/Volumes/Untitled/Data/'
SEG_DIR = os.path.join(BASE_DIR, 'results/segs/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'statistics/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output log file for diagnostics
LOG_FILE = os.path.join(OUTPUT_DIR, f'cr{CR}_detection_log.txt')

def log_message(msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + '\n')

# Respike
def respike_data(A, spikes):
    # Add spikes back to AIA data
    spikes_array = np.asarray(spikes)
    
    if spikes_array.shape[0] == 3:
        spikes_to_process = spikes_array
    elif spikes_array.shape[1] == 3:
        spikes_to_process = spikes_array.T
    else:
        return A
    
    for s in np.transpose(spikes_to_process):
        idx = int(s[0])
        n, m = np.unravel_index(idx, A.shape)
        A[n, m] = s[1]
    return A

# Functions
def compute_signed_distance(seg_mask):
    # Distance from boundary
    return (ndimage.distance_transform_edt(~seg_mask) - 
            ndimage.distance_transform_edt(seg_mask) + seg_mask - 0.5)

def process_frame_pair(I1, I2, seg, min_area, max_dist):
    # Frame pair process
    
    D = I2 - I1
    dist_map = compute_signed_distance(seg)
    dist_mask = (dist_map < max_dist) & (dist_map > -max_dist)
    
    # Positive flashes
    pos_pixels = D[D > 0]
    if len(pos_pixels) > 0:
        T_pos = pos_pixels.mean() + pos_pixels.std()
        pos_flashes = D > T_pos
    else:
        pos_flashes = np.zeros_like(D, dtype=bool)
    
    # Negative flashes
    neg_pixels = D[D < 0]
    if len(neg_pixels) > 0:
        T_neg = neg_pixels.mean() - neg_pixels.std()
        neg_flashes = D < T_neg
    else:
        neg_flashes = np.zeros_like(D, dtype=bool)
    
    # Apply distance mask
    pos_flashes = morphology.remove_small_objects(pos_flashes & dist_mask, min_area)
    neg_flashes = morphology.remove_small_objects(neg_flashes & dist_mask, min_area)
    
    return pos_flashes, neg_flashes, dist_map

def extract_flash_properties_fast(pos_labeled, dist_map, original_image, max_flashes=None):
    from skimage.measure import regionprops
    
    props = regionprops(pos_labeled, intensity_image=original_image)
    
    if max_flashes is not None and len(props) > max_flashes:
        props = props[:max_flashes]
    
    n = len(props)
    if n == 0:
        return [], [], [], []
    
    areas = np.zeros(n, dtype=np.int32)
    distances = np.zeros(n, dtype=np.float32)
    intensities = np.zeros(n, dtype=np.float32)
    centroids = np.zeros((n, 2), dtype=np.float32)
    
    for i, prop in enumerate(props):
        areas[i] = prop.area
        centroids[i, 0] = prop.centroid[1]
        centroids[i, 1] = prop.centroid[0]
        intensities[i] = prop.mean_intensity
        
        coords = prop.coords
        distances[i] = np.mean(dist_map[coords[:, 0], coords[:, 1]])
    
    return areas, distances, intensities, centroids

# Locates correct file
def find_data_path(cr):
    possible_paths = [
        os.path.join(DATA_DIR, f'CR{cr}_12sec/'),
        os.path.join(DATA_DIR, f'CR{cr}/'),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def get_file_lists_simple(data_path):
    image_files = []
    spike_files = []
    
    direct_images = sorted(glob.glob(os.path.join(data_path, '*.image_lev1.fits')))
    direct_spikes = sorted(glob.glob(os.path.join(data_path, '*.spikes.fits')))
    
    if len(direct_images) > 0:
        return direct_images, direct_spikes
    
    subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for subdir in subdirs:
        subpath = os.path.join(data_path, subdir)
        images = sorted(glob.glob(os.path.join(subpath, '*.image_lev1.fits')))
        spikes = sorted(glob.glob(os.path.join(subpath, '*.spikes.fits')))
        
        if len(images) > 0:
            image_files.extend(images)
            spike_files.extend(spikes)
    
    return sorted(image_files), sorted(spike_files)

# Main function
def run_full_detection():
    respike_str = 'respike1' if RESPIKE else 'respike0'
    
    log_message("="*60)
    log_message(f"FULL FLASH DETECTION")
    log_message(f"CR{CR} - {'RESPIKED' if RESPIKE else 'DESPIKED'}")
    log_message(f"All frames, all flashes (NO LIMITS)")
    log_message("="*60)
    
    # Find data path
    data_path = find_data_path(CR)
    if data_path is None:
        log_message(f"ERROR: No data found for CR{CR}")
        return None
    
    log_message(f"Data path: {data_path}")
    
    # Get files
    image_files, spike_files = get_file_lists_simple(data_path)
    log_message(f"Image files: {len(image_files)}")
    
    if len(image_files) == 0:
        log_message("ERROR: No image files found!")
        return None
    
    # Segmentation files
    seg_cr_dir = os.path.join(SEG_DIR, f'CR{CR}/')
    seg_files = sorted(glob.glob(os.path.join(seg_cr_dir, '*.tiff')))
    
    if len(seg_files) == 0:
        log_message(f"ERROR: No segmentation files found in {seg_cr_dir}")
        return None
    
    log_message(f"Segmentation files: {len(seg_files)}")
    
    # Process all frames
    n_frames = min(len(image_files), len(seg_files) * 2)
    log_message(f"Processing {n_frames-1} frame pairs")
    
    # Storage
    all_areas = []
    all_distances = []
    all_intensities = []
    all_centroids = []
    
    start_time = time.time()
    frames_with_flashes = 0
    last_log_time = start_time
    
    for idx in range(1, n_frames):
        # Log progress every 100 frames
        if idx % 100 == 0 or idx == n_frames - 1:
            elapsed = time.time() - start_time
            fps = frames_with_flashes / elapsed if elapsed > 0 else 0
            log_message(f"  Frame {idx}/{n_frames-1} (elapsed: {elapsed/60:.1f}m, flashes: {len(all_areas)}, fps: {fps:.1f})")
        
        try:
            if idx >= len(seg_files):
                break
            
            # Load segmentation
            seg = np.array(Image.open(seg_files[seg_idx])) > 0
            
            # Load images
            with fits.open(image_files[idx-1]) as hdulist:
                I1 = hdulist[1].data.astype(np.float32)
            with fits.open(image_files[idx]) as hdulist:
                I2 = hdulist[1].data.astype(np.float32)
            
            # Respike if enabled
            if RESPIKE:
                if spike_files and idx-1 < len(spike_files):
                    try:
                        I1 = respike_data(I1, fits.getdata(spike_files[idx-1]))
                    except:
                        pass
                if spike_files and idx < len(spike_files):
                    try:
                        I2 = respike_data(I2, fits.getdata(spike_files[idx]))
                    except:
                        pass
            
            # Detect flashes
            pos_flashes, neg_flashes, dist_map = process_frame_pair(
                I1, I2, seg, MIN_AREA, MAX_DIST)
            
            # Label positive flashes
            if pos_flashes.sum() > 0:
                pos_labeled = measure.label(pos_flashes)
                
                areas, distances, intensities, centroids = extract_flash_properties_fast(
                    pos_labeled, dist_map, I2, MAX_FLASHES_PER_FRAME)
                
                if len(areas) > 0:
                    all_areas.extend(areas)
                    all_distances.extend(distances)
                    all_intensities.extend(intensities)
                    all_centroids.extend(centroids)
                    frames_with_flashes += 1
            
            # Clear memory
            del I1, I2, seg, pos_flashes, neg_flashes, dist_map
            gc.collect()
            
        except Exception as e:
            log_message(f"    Error at frame {idx}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Convert to arrays
    areas = np.array(all_areas) if all_areas else np.array([])
    distances = np.array(all_distances) if all_distances else np.array([])
    intensities = np.array(all_intensities) if all_intensities else np.array([])
    centroids = np.array(all_centroids) if all_centroids else np.array([])
    
    # Summary
    log_message("\n" + "="*60)
    log_message("DETECTION COMPLETE")
    log_message("="*60)
    log_message(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    log_message(f"Frames processed: {n_frames-1}")
    log_message(f"Frames with flashes: {frames_with_flashes}")
    log_message(f"Total flashes: {len(areas)}")
    log_message(f"Mean area: {np.mean(areas) if len(areas) > 0 else 0:.2f} pixels")
    log_message(f"Mean distance: {np.mean(distances) if len(distances) > 0 else 0:.2f} pixels")
    log_message(f"Mean intensity: {np.mean(intensities) if len(intensities) > 0 else 0:.2f}")
    
    # Save stats
    stats = {
        'cr': CR,
        'respike': RESPIKE,
        'n_frames': n_frames - 1,
        'total_flashes': len(areas),
        'areas': areas,
        'distances': distances,
        'intensities': intensities,
        'centroids': centroids,
        'mean_area': np.mean(areas) if len(areas) > 0 else 0,
        'std_area': np.std(areas) if len(areas) > 0 else 0,
        'median_area': np.median(areas) if len(areas) > 0 else 0,
        'mean_distance': np.mean(distances) if len(distances) > 0 else 0,
        'std_distance': np.std(distances) if len(distances) > 0 else 0,
        'median_distance': np.median(distances) if len(distances) > 0 else 0,
        'mean_intensity': np.mean(intensities) if len(intensities) > 0 else 0,
        'std_intensity': np.std(intensities) if len(intensities) > 0 else 0,
        'inside_ch': np.sum(distances < 0) if len(distances) > 0 else 0,
        'outside_ch': np.sum(distances > 0) if len(distances) > 0 else 0,
        'at_boundary': np.sum(np.abs(distances) <= 1) if len(distances) > 0 else 0,
    }
    
    output_file = os.path.join(OUTPUT_DIR, f'cr{CR}_statistics_{respike_str}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(stats, f)
    
    log_message(f"\n✓ Statistics saved to: {output_file}")
    log_message(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    log_message("="*60)
    log_message("DETECTION COMPLETE - YOU CAN SHUT DOWN YOUR COMPUTER NOW")
    
    return stats

# Code execution
if __name__ == "__main__":
    stats = run_full_detection()
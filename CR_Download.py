# Joshua Johnson
# April 8, 2026
# Downloads CRs from SDO/AIA Database (12-secon cadence for 30-minute periods each day per CR)

import pandas as pd
import glob
import numpy as np
import os
import datetime
import drms
import time
import copy
import random

client = drms.Client()


CarringtonFile = 'Carrington Rotation Start Dates.csv'
StartRotation  = 2099 # CR start
EndRotation    = 2133 # CR end
RotCadence     = 1

# Dataset folder
dataFolder  = '/Volumes/Untitled/Data'
traceFolder = 'DownloadLists/'

# Request Data
AIAserise  = 'aia.lev1_euv_12s'
notify     = 'joshua27@nmsu.edu'
AIAsegment = 'image,spikes'
reckeys    = 'T_REC, QUALITY'
protocol   = 'fits'

# 12-second cadence settings
cadence_seconds = 12
window_minutes = 15
images_per_day = int((window_minutes * 2 * 60) / cadence_seconds)  # 150

# Each request returns 4 files (2 images + 2 spikes)
expected_files_per_request = 4

# Buffer time settings
min_request_interval = 2
max_request_interval = 4
batch_size = 20
batch_pause = 10
day_completion_pause = 15
sleepTime = 30

verbose = True # Progress log
cycle = 0
cycleLimit = 5

# Helper Functions
def timeFormat(dt):
    return dt.strftime('%Y.%m.%d_%H:%M:%S')

# Read File
data = pd.read_csv(CarringtonFile, header=0)
keys = data.keys()
startIndex = np.where(data[keys[0]] == StartRotation)[0][0]

# Cycle Through Data
for i in range(startIndex, len(data)-1, RotCadence):
    RotNum = data[keys[0]][i]
    
    if RotNum > EndRotation:
        break
    
    if verbose:
        print('\n' + '='*80)
        print(f'Rotation: {RotNum}')
        print('='*80)
    
    year = data[keys[1]][i]
    month = data[keys[2]][i]
    day = int(data[keys[3]][i])
    hour = int((data[keys[3]][i] - day) * 24)
    startDate = datetime.datetime(int(year), int(month), day, hour, 0, 0)
    
    year = data[keys[1]][i+1]
    month = data[keys[2]][i+1]
    day = int(data[keys[3]][i+1])
    hour = int((data[keys[3]][i+1] - day) * 24)
    minute = int((((data[keys[3]][i+1] - day) * 24) - hour) * 60)
    second = int((((((data[keys[3]][i+1] - day) * 24) - hour) * 60) - minute) * 60)
    endDate = datetime.datetime(int(year), int(month), day, hour, minute, second)
    
    if endDate < datetime.datetime.now():
        traceName = os.path.join(traceFolder, f'CR{RotNum}_12sec.csv')
        os.makedirs(traceFolder, exist_ok=True)
        
        if not os.path.exists(traceName):
            with open(traceName, 'w+') as f:
                f.write('T_intended,T_REC,folder_label,image_file,spike_file,download_time\n')
        
        crFolder = os.path.join(dataFolder, f'CR{RotNum}_12sec')
        os.makedirs(crFolder, exist_ok=True)
        
        date = copy.deepcopy(startDate)
        max_days = 7
        day_counter = 0
        
        while date <= endDate and day_counter < max_days:
            base_time = datetime.datetime(date.year, date.month, date.day, 12, 0, 0)
            
            if verbose:
                print(f'\n--- Processing {base_time.strftime("%Y-%m-%d")} ---')
                print(f'    Generating {images_per_day} images at {cadence_seconds}-second cadence')
            
            request_counter = 0
            
            for image_idx in range(images_per_day):
                image_time = base_time + datetime.timedelta(minutes=-window_minutes, 
                                                           seconds=image_idx * cadence_seconds)
                
                try:
                    buffer_seconds = 6
                    beg_date = timeFormat(image_time + datetime.timedelta(seconds=-buffer_seconds)) + 'Z'
                    end_date = timeFormat(image_time + datetime.timedelta(seconds=buffer_seconds)) + 'Z'
                    T_intended = timeFormat(image_time) + 'Z'
                    
                    AIArequest = f'{AIAserise}[{beg_date}-{end_date}][193]'
                    
                    if verbose:
                        print(f'    [{image_idx+1:3d}/{images_per_day}] Time: {image_time.strftime("%H:%M:%S")}')
                    
                    AIAquery = client.query(AIArequest, seg=AIAsegment, key=reckeys)
                    
                    if len(AIAquery[0]) == 0:
                        if verbose:
                            print(f'        No data found')
                        continue
                    
                    T_REC = AIAquery[0][reckeys.split(', ')[0]][0]
                    QUALITY = AIAquery[0][reckeys.split(', ')[1]]
                    
                    outDir = T_REC.replace(':', '').replace('.', '').replace('-', '')
                    outDir_path = os.path.join(crFolder, outDir)
                    os.makedirs(outDir_path, exist_ok=True)
                    
                    # FIXED: Check if we have at least some files (not exactly 300)
                    files = glob.glob(os.path.join(outDir_path, '*.fits'))
                    
                    # Only download if no files exist (to avoid re-downloading)
                    if len(files) == 0:
                        if len(QUALITY) == 2 and np.max(QUALITY) == 0:
                            if verbose:
                                print(f'        Good quality, downloading...')
                            
                            AIArequest_full = AIArequest + '{' + AIAsegment + '}'
                            result = client.export(AIArequest_full, method='url', 
                                                protocol=protocol, email=notify)
                            result.download(outDir_path)
                            
                            # Update trace file
                            files = glob.glob(os.path.join(outDir_path, '*.fits'))
                            if len(files) >= 2:  # At least one image+spike pair
                                image_file = 'EMPTY'
                                spike_file = 'EMPTY'
                                for file in files:
                                    if 'image' in os.path.basename(file):
                                        image_file = outDir + '/' + os.path.basename(file)
                                    elif 'spikes' in os.path.basename(file):
                                        spike_file = outDir + '/' + os.path.basename(file)
                                
                                row = f"{T_intended},{T_REC},{outDir},{image_file},{spike_file},{datetime.datetime.now()}\n"
                                with open(traceName, 'a+') as f:
                                    f.write(row)
                                
                                if verbose:
                                    print(f'        ✓ Saved ({len(files)} files)')
                            else:
                                print(f'        ✗ Download had issues')
                        else:
                            if verbose:
                                print(f'        ✗ Quality error')
                    else:
                        if verbose:
                            print(f'        Files already exist, skipping')
                    
                    # Rate limiting
                    request_counter += 1
                    buffer_time = random.uniform(min_request_interval, max_request_interval)
                    time.sleep(buffer_time)
                    
                    if request_counter % batch_size == 0 and request_counter < images_per_day:
                        if verbose:
                            print(f'        Batch pause {batch_pause}s...')
                        time.sleep(batch_pause)
                    
                except Exception as e:
                    if verbose:
                        print(f'        Error: {e}')
                    
                    if cycle > cycleLimit:
                        cycle = 0
                        continue
                    else:
                        cycle += 1
                        time.sleep(sleepTime)
            
            date += datetime.timedelta(days=1)
            day_counter += 1
            
            if verbose and day_counter < max_days:
                print(f'\n    Day completed. Pausing {day_completion_pause}s...')
            time.sleep(day_completion_pause)
        
        if verbose:
            print(f'\nCompleted rotation {RotNum}')

print('\n' + '='*80)
print('**Process Complete**')
print('='*80)
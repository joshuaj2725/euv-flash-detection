# Joshua Johnson
# April 8, 2026
# Generates CSV files for CR download

import pandas as pd
from sunpy.coordinates.sun import carrington_rotation_time
import astropy.units as u

rotations = [2099, 2106, 2113, 2133]
data = []

for rot in rotations:
    start_time = carrington_rotation_time(rot)
    
    start_date = start_time.to_datetime()
    
    year = start_date.year
    month = start_date.month

    start_date_float = start_date.day + (start_date.hour / 24.0) + (start_date.minute / 1440.0)
    
    data.append({
        'Rotation': rot,
        'Year': year,
        'Month': month,
        'StartDate': round(start_date_float, 6)
    })
    
    print(f'CR{rot}: {start_date.strftime("%Y-%m-%d %H:%M:%S")} (StartDate: {round(start_date_float, 6)})')

df = pd.DataFrame(data)
df.to_csv('Carrington Rotation Start Dates.csv', index=False)

print('\nFile contents:')
print(df.to_string(index=False))
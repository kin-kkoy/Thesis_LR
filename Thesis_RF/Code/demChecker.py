import rasterio
import numpy as np

# List of files you want to check
files_to_check = ['lapulapuDEM.tif']

for filename in files_to_check:
    try:
        with rasterio.open(filename) as src:
            print(f"--- Checking {filename} ---")
            print(f"Bands: {src.count}")
            print(f"Data Type: {src.dtypes[0]}")
            
            # Read the first band to check values
            data = src.read(1)
            # Mask out 'no data' values if they exist
            if src.nodata is not None:
                data = data[data != src.nodata]
                
            print(f"Min Value: {data.min()}")
            print(f"Max Value: {data.max()}")
            
            # Judgment Logic
            if src.count > 1:
                print("❌ VERDICT: Unusable (It is an RGB image, not raw data)")
            elif "int" in src.dtypes[0] and data.max() > 360:
                 # Rough heuristic: if integer and huge values, might be scaled or raw counts
                 print("⚠️ VERDICT: Warning (Check if units are correct)")
            else:
                 print("✅ VERDICT: Likely Usable (appears to be single-band data)")
            print("\n")
            
    except Exception as e:
        print(f"Could not open {filename}: {e}")
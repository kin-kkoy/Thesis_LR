import numpy as np
import rasterio

def calculate_slope_aspect(dem_path, slope_out_path, aspect_out_path):
    with rasterio.open(dem_path) as src:
        # 1. Read the DEM data (height in meters)
        dem = src.read(1).astype('float32')
        
        # 2. Get pixel size (resolution) in meters
        # This is crucial: we need to know that 1 pixel = 3 meters
        x_res = src.transform[0]
        y_res = -src.transform[4]  # Usually negative in GeoTIFFs
        
        print(f"Processing DEM with resolution: {x_res}m x {y_res}m")

        # 3. Calculate Gradients (Change in height per pixel)
        # np.gradient returns [gradient_y, gradient_x]
        # We divide by resolution to get "rise over run" in meters
        dy, dx = np.gradient(dem, y_res, x_res)

        # 4. Calculate Slope
        # Formula: arctan(sqrt(dx^2 + dy^2)) * (180 / pi)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # 5. Calculate Aspect
        # Formula: arctan2(-dy, dx) adjusted to 0-360 degrees (North=0)
        # Note: Math convention is usually CCW from East, Geography is CW from North.
        # This standard conversion handles the rotation:
        aspect_rad = np.arctan2(-dy, dx)
        aspect_deg = np.degrees(aspect_rad)
        
        # Convert to compass bearing (0 is North, 90 is East)
        # The result of the math above needs this adjustment:
        aspect_deg = (90 - aspect_deg) % 360

        # 6. Save Outputs
        # We use the metadata from the original file but update data type
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=-9999)

        # Write Slope
        with rasterio.open(slope_out_path, 'w', **profile) as dst:
            dst.write(slope_deg.astype(rasterio.float32), 1)
            print(f"✅ Slope saved to: {slope_out_path}")

        # Write Aspect
        with rasterio.open(aspect_out_path, 'w', **profile) as dst:
            dst.write(aspect_deg.astype(rasterio.float32), 1)
            print(f"✅ Aspect saved to: {aspect_out_path}")

try:
    calculate_slope_aspect(
        dem_path="3x3DEM.tif", 
        slope_out_path="second_slope.tif", 
        aspect_out_path="second_aspect.tif"
    )
except FileNotFoundError:
    print("❌ Error: Could not find your input DEM file. Make sure the path is correct.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
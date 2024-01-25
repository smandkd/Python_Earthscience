#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cartopy.crs as crs
import pandas as pd

# %%
dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
# IBTrACS 2004 - 2020
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp') & (filtered_DATA.usa_wind >= 34), drop=True)
# IBTrACS USA AGENCY JTWC_WP
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True)
# Haitang 
haitang_GEN_index = 0
lmi = tc_HAITANG.usa_wind.max()
haitang_LMI_index = np.where(tc_HAITANG.usa_wind == lmi)[1][-1]
haitang_GEN2LMI = tc_HAITANG.isel(date_time=slice(haitang_GEN_index, haitang_LMI_index+1))
haitang_GEN2LMI # GEN to LMI 

# %%
haitang_GEN2LMI

# %%
tc_HAITANG_00 = haitang_GEN2LMI.where(haitang_GEN2LMI['time'][0].time.dt.hour == 0, drop=True)
tc_HAITANG_00
# Hour : 00 

# %%
days_3days_before_revised = tc_HAITANG_00['time'] - pd.Timedelta(days=3)
days_3days_before_revised 
# Substraction 3 days in  variable 'time' 
dt_array = []
for i in range(len(days_3days_before_revised[0])):
    datetime = np.array(days_3days_before_revised[0][i], dtype='datetime64[ns]')
    time = np.datetime64(datetime, 'h')
    dt_array.append(time)
# Extracting variable 'time' 
lat = np.array(tc_HAITANG_00.usa_lat[0], dtype=object)
lon = np.array(tc_HAITANG_00.usa_lon[0], dtype=object)
# Haitang_00 usa latitude, longitude 
HAITANG_00_coords_with_days3before = np.column_stack((lat,lon))
HAITANG_00_coords_with_days3before
# Coordinates of Haitang_00 
# %%
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')
days_3_before_oisst = oisst_DATA.sel(time=dt_array, drop=True)
"""
Extracting data align with 
2005-07-09T00
2005-07-10T00
2005-07-11T00
2005-07-12T00
2005-07-13T00
2005-07-14T00
in variable time
"""

# Points 
# %%    
plt.figure(figsize=(10,5))
ax = plt.axes(projection=crs.PlateCarree())
ax.coastlines(resolution='10m')
ax.set_extent([100, 155, 10, 48])
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

for index, t in enumerate(dt_array):
    sst_sum = 0
    oisst_07 = oisst_DATA.sel(time=t)
    oisst_07_lat = oisst_07.lat
    oisst_07_lon = oisst_07.lon - 180
    oisst_07_sst = oisst_07.sst.data.flatten()
    oisst_07_coords = np.array(np.meshgrid(oisst_07_lat, oisst_07_lon)).T.reshape(-1, 2)
    oisst_07_coords_tree = cKDTree(oisst_07_coords)
    oisst_07_coords_sst = np.column_stack((oisst_07_coords, oisst_07_sst))
    
    haitang_07_coord = HAITANG_00_coords_with_days3before[index]
    thres = 200/111

    indices = oisst_07_coords_tree.query_ball_point(haitang_07_coord, thres)
    points_07 = oisst_07_coords_sst[indices]
    
    for j in range(len(points_07)):
        ax.plot(points_07[j][1], points_07[j][0], color="red", marker='o', markersize=0.5, transform=crs.PlateCarree())
        sst_sum += points_07[j][2]
        
    print(f'{t} mean : {(sst_sum)/len(points_07)}')
    print(f'{t} points count : {len(points_07)}')
    print(f'{t} center coord : {haitang_07_coord[1]}, {haitang_07_coord[0]}')
    print('\n')
    
plt.savefig('every_points_nearby_haitangcenter.pdf')
plt.show()

# %%

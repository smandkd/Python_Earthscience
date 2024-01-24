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
# drop = True 로 할 경우 결측값을 결과 배열에서 완전히 제거한다. 
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp') & (filtered_DATA.usa_wind >= 34), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True)
GEN_index = np.where(tc_HAITANG.usa_wind >= 34)[1][0]
LMI_index = np.where(tc_HAITANG.usa_wind == tc_HAITANG.usa_wind.max())[1][-1]
HAITANG_gen2lmi_lat = tc_HAITANG.usa_lat[0][GEN_index:LMI_index+1]
HAITANG_gen2lmi_lon = tc_HAITANG.usa_lon[0][GEN_index:LMI_index+1]
HAITANG_coord = np.array(list(zip(HAITANG_gen2lmi_lat, HAITANG_gen2lmi_lon)))
HAITANG_gen2lmi = tc_HAITANG.isel(date_time=slice(GEN_index, LMI_index+1), drop=True)
HAITANG_gen2lmi
hr_00_data = HAITANG_gen2lmi.where(HAITANG_gen2lmi.time.dt.hour == 00, drop=True)
lat = np.array(hr_00_data.usa_lat[0], dtype=object)
lon = np.array(hr_00_data.usa_lon[0], dtype=object)
days_3_before = hr_00_data['time'] - pd.Timedelta(days=3)
# %%
HAITANG_gen2lmi[0]
# %%

time_array = np.array(days_3_before)
date_array = pd.to_datetime(time_array[0]).floor('H')
# 시간 단위 내림
# %%
HAITANG_00_coords_with_days3before = np.column_stack((lat,lon,pd.to_datetime(date_array)))
HAITANG_00_coords_with_days3before[1][2]
# %%
pd.to_datetime(HAITANG_00_coords_with_days3before[1][2])
# %%
OISST_dataset = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')
OISST_dataset
# %%
oisst_0710 = OISST_dataset.sel(time=pd.to_datetime(HAITANG_00_coords_with_days3before[1][2]))
oisst_0710
# oisst 에서 0710에 해당하는 데이터들 
# %%


# %%
plt.figure(figsize=(10, 5))
ax = plt.axes(projection=crs.PlateCarree())
ax.coastlines(resolution='10m')
ax.set_extent([100, 155, 10, 40])
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

for i in range(len(HAITANG_00_coords_with_days3before)):
    oisst_haitang_3daysbefore = OISST_dataset.sel(time=pd.to_datetime(HAITANG_00_coords_with_days3before[i][2]))
    oisst_LAT = oisst_haitang_3daysbefore.lat
    oisst_LON = oisst_haitang_3daysbefore.lon - 180
    sst = oisst_haitang_3daysbefore.sst.data.flatten()
    oisst_coords = np.array( np.meshgrid( oisst_LAT, oisst_LON ) ).T.reshape(-1, 2) 
    oisst_coords_with_sst = np.column_stack( (oisst_coords ,sst) )
    haitang_LAT = HAITANG_00_coords_with_days3before[i][0]
    haitang_LON = HAITANG_00_coords_with_days3before[i][1]
    haitang_COORD_i = [haitang_LAT, haitang_LON]
    
    oisst_COORDS_tree = cKDTree(oisst_coords)
    distance_THRESHOLD = 200/111
    indices = oisst_COORDS_tree.query_ball_point(haitang_COORD_i, distance_THRESHOLD)
    near_points = oisst_coords_with_sst[indices]
    mean = 0
    for j in range(len(near_points)):
        plt.scatter(x=near_points[j][1], y=near_points[j][0], color="red", s=1, alpha=0.5, transform=crs.PlateCarree())
        mean += near_points[j][2]
        
    print(f"{pd.to_datetime(HAITANG_00_coords_with_days3before[i][2])} sst average : {mean/len(near_points)}")
    
plt.savefig("HAITANG_3daysago_sstaverage_and_plot.pdf")
plt.show()


# %%
len(HAITANG_00_coords_with_days3before)
print(f"{pd.to_datetime(HAITANG_00_coords_with_days3before[i][2])} : {mean/len(near_points[0])}")
"""
Haitang (2005)
태풍에 대한 각 경로 포인트 주위 200km 반경 내의 MSL 값의 평균. 
"""

#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd

#%%
def lmi(wind_array):
    if np.all(np.isnan(wind_array)):
        return None
    max_value = np.nanmax(wind_array)
    if np.isnan(max_value):
        return None
    
    lmi_index = np.where(wind_array == max_value)[0]
    
    if lmi_index.size > 0:
        return lmi_index[0]
    else:
        return None    

def gen(wind_array):
    gen_index = np.where(wind_array >= 34)[0]
    
    if gen_index.size > 0:
        return gen_index[0]
    else:
        return None    
#%%
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp'), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True) # Haitang 2005

#%%
wind_array = tc_HAITANG.usa_wind.data[0]
gen_index = gen(wind_array)
lmi_index = lmi(wind_array)
selected_indices = np.arange(gen_index, lmi_index+1)
haitang_dataset = tc_HAITANG.isel(date_time=selected_indices, drop=True)

#%%
haitang_dataset.usa_wind.data


# %%
days_3days_before_revised = haitang_dataset['time'] - pd.Timedelta(days=3)
haitang_time = days_3days_before_revised['time']
haitang_time


#%%
# Substraction 3 days in  variable 'time' 
dt_array = []
for data in haitang_time[0]:
    datetime = np.array(data, dtype='datetime64[ns]')
    time = np.datetime64(datetime, 'D')
    dt_array.append(time)

#%%

dt_array

#%%
lat = np.array(haitang_dataset.usa_lat[0], dtype=object)
lon = np.array(haitang_dataset.usa_lon[0], dtype=object)
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lat,lon))
HAITANG_coords
# Coordinates of Haitang
#%%

print(len(dt_array)) # 20 개

#%%

dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
dataset_era5_mslp
# %%

days_3_before_mslp = dataset_era5_mslp.sel(time=dt_array, drop=True)

#%%

days_3_before_mslp

# %%
def open_mslp_dataset(index):
    data = days_3_before_mslp.isel(time=index, drop=True)
    return data
# %%

dis_thres = 200/111

mslp_daily_averages = []

for index, time in enumerate(dt_array):
    mslp_dataset = open_mslp_dataset(index)
    
    mslp_lat = mslp_dataset.latitude
    mslp_lon = mslp_dataset.longitude
    mslp_value = mslp_dataset.msl.data.flatten()
    mslp_coords = np.array(np.meshgrid(mslp_lat, mslp_lon)).T.reshape(-1, 2)
    mslp_tree = cKDTree(mslp_coords)
    coords_msl = np.column_stack((mslp_coords, mslp_value))
    haitang_index_coord = HAITANG_coords[index]

    indices = mslp_tree.query_ball_point(haitang_index_coord, dis_thres)
    msl_200_points = coords_msl[indices]
    msl = msl_200_points[:, -1]
    msl_ave = np.mean(msl)
    mslp_daily_averages.append(msl_ave)
    
    
for i, avg in enumerate(mslp_daily_averages):
    print(f'{dt_array[i]} : {avg}\n')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# %%

"""
Haitang (2005)
태풍에 대한 각 경로 포인트 주위 200km 반경 내의 Air Temperature 값의 평균. 
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

# %%
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
airt_dataset
# Temperature unit is K.
# %%
days_3_before_airt = airt_dataset.sel(time=dt_array, drop=True)
days_3_before_airt
# %%
def open_airt_dataset(index):
    data = days_3_before_airt.isel(time=index, drop=True)
    return data

def open_level_dataset(level, dataset):
    dataset = dataset.isel(level = level, drop=True)
    return dataset

def filtered_airt(data, lat, lon):
    result = data.sel(latitude=lat, longitude=lon, drop=True)
    return result
# %%
dis_thres = 200/111

array = []

for index, time in enumerate(dt_array):
    level_airt_arr = []
    
    airt_dataset = open_airt_dataset(index)
    lat = airt_dataset.latitude
    lon = airt_dataset.longitude
    airt_coords = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)
    airt_tree = cKDTree(airt_coords)
    haitang_coord = HAITANG_coords[index]
    
    indices = airt_tree.query_ball_point(haitang_coord, dis_thres)
    
    airt_200_points = airt_coords[indices]
    
    lat = airt_200_points[:,0]
    lon = airt_200_points[:,-1]
    new_airt_data = filtered_airt(airt_dataset, lat, lon)
    
    for level in range(len(airt_dataset.level.data)):
        dataset = open_level_dataset(level, new_airt_data)
        airt_arr = dataset.t.data
        airt_mean = np.mean(airt_arr)
        level_airt_arr.append(airt_mean)
        
    array.append(level_airt_arr)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
# %%
array
# %%

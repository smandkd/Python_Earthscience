"""
Haitang (2005)
태풍에 대한 각 경로 포인트 주위 200km 반경 내의 SST 값의 평균. 
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
# %%
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
lat = np.array(haitang_dataset.usa_lat[0], dtype=object)
lon = np.array(haitang_dataset.usa_lon[0], dtype=object)
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lat,lon))
HAITANG_coords
# Coordinates of Haitang
#%%

print(len(dt_array)) # 20 개

# %%
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')

#%%
oisst_DATA.isel(time=1,drop=True)

#%%
days_3_before_oisst = oisst_DATA.sel(time=dt_array, drop=True)
"""
Extracting data align with 
[numpy.datetime64('2005-07-08'),
 numpy.datetime64('2005-07-09'),
 numpy.datetime64('2005-07-09'),
 numpy.datetime64('2005-07-09'),
 numpy.datetime64('2005-07-09'),
 numpy.datetime64('2005-07-10'),
 numpy.datetime64('2005-07-10'),
 numpy.datetime64('2005-07-10'),
 numpy.datetime64('2005-07-10'),
 numpy.datetime64('2005-07-11'),
 numpy.datetime64('2005-07-11'),
 numpy.datetime64('2005-07-11'),
 numpy.datetime64('2005-07-11'),
 numpy.datetime64('2005-07-12'),
 numpy.datetime64('2005-07-12'),
 numpy.datetime64('2005-07-12'),
 numpy.datetime64('2005-07-12'),
 numpy.datetime64('2005-07-13'),
 numpy.datetime64('2005-07-13'),
 numpy.datetime64('2005-07-13')]
in variable time

Dim -> 
time: 20, lat: 720, lon: 1440
"""
#%%

days_3_before_oisst

# %%
def open_oisst_dataset(index):
    data = oisst_DATA.isel(time=index, drop=True)
    return data
    
    
# %%
dis_thres = 200/111

sst_daily_averages = []

for index, time in enumerate(dt_array):
    print(f'{time} {index}')
    oisst_dataset = open_oisst_dataset(index)

    oisst_lat = oisst_dataset.lat
    oisst_lon = oisst_dataset.lon - 180
    oisst_sst = oisst_dataset.sst.data.flatten()
    oisst_coords = np.array(np.meshgrid(oisst_lat, oisst_lon)).T.reshape(-1, 2)
    oisst_tree = cKDTree(oisst_coords)
    oisst_coords_sst = np.column_stack((oisst_coords, oisst_sst))
    haitang_index_coord = HAITANG_coords[index]
    
    indices = oisst_tree.query_ball_point(haitang_index_coord, dis_thres)
    oisst_200_points = oisst_coords_sst[indices]
    sst_average = np.mean(oisst_200_points[:, -1])  # 마지막 열(SST)에 대한 평균
    sst_daily_averages.append(sst_average)
    
    print(f'SST Average for {time}: {sst_average}')
    print('---------------------------------------------')

# 최종적으로 각 일자별 평균 SST 출력
print("Daily SST Averages:")
for i, avg in enumerate(sst_daily_averages):
    print(f'{dt_array[i]}: {avg}')
# %%

# -----------------------------------------
# MPI Input data
# var : airt(Air Temperature), shum(Specific Humidity), mslp(Mean sea level pressure), sst(Sea Surface Temperature)  
# dims : time, level
# -----------------------------------------
#%%
import xarray as xr
import pickle
from pyPI import pi
from pyPI.utilities import *
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

def isel_time_dataset(data, index):
    result = data.isel(time=index, drop=True)
    return result

def isel_level_dataset(data, index):
    result = data.isel(level=index, drop=True)
    return result

def sel_lat_lon_dataset(data, lat, lon):
    result = data.sel(latitude=lat, longitude=lon, drop=True)
    return result
#%%
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp'), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True) # Haitang 2005

wind_array = tc_HAITANG.usa_wind.data[0]
gen_index = gen(wind_array)
lmi_index = lmi(wind_array)
selected_indices = np.arange(gen_index, lmi_index+1)
haitang_dataset = tc_HAITANG.isel(date_time=selected_indices, drop=True)

days_3days_before_revised = haitang_dataset['time'] - pd.Timedelta(days=3)
haitang_time = days_3days_before_revised['time']
haitang_time
# Substraction 3 days in  variable 'time' 
dt_array = []
for data in haitang_time[0]:
    datetime = np.array(data, dtype='datetime64[ns]')
    time = np.datetime64(datetime, 'D')
    dt_array.append(time)

lat = np.array(haitang_dataset.usa_lat[0], dtype=object)
lon = np.array(haitang_dataset.usa_lon[0], dtype=object)
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lat,lon))
HAITANG_coords
#%%
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')

dis_thres = 200/111

before_3_shum_dataset = shum_dataset.sel(time=dt_array, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=dt_array, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=dt_array, drop=True)
days_3_before_airt = airt_dataset.sel(time=dt_array, drop=True)
#%%
level_arr = days_3_before_airt.level.data
level_arr

#%%
# -------------------
#
# Specific Humidity
#
# -------------------
shum_mean_arr = []

for index, time in enumerate(dt_array):
    shum_arr = []
    
    shum_dataset = isel_time_dataset(before_3_shum_dataset, index)
    
    lat = shum_dataset.latitude
    lon = shum_dataset.longitude
    shum_coords = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)
    shum_tree = cKDTree(shum_coords)
    haitang_coord = HAITANG_coords[index]
    
    indices = shum_tree.query_ball_point(haitang_coord, dis_thres)
    
    shum_200_points = shum_coords[indices]
    
    new_lat = shum_200_points[:, 0]
    new_lon = shum_200_points[:, -1]
    len_level = len(shum_dataset.level.data)
    filtered_shum_dataset = sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
    
    for level in range(len_level):
        dataset = isel_level_dataset(filtered_shum_dataset, level)
        
        shum = dataset.q.data
        shum_mean = np.mean(shum)
        shum_arr.append(shum_mean)
        
        
    shum_mean_arr.append(shum_arr)
#%%
# -------------------------
#
# Sea Sureface Temperature
#
# -------------------------
sst_daily_averages = []

for index, time in enumerate(dt_array):
    oisst_dataset = isel_time_dataset(days_3_before_oisst, index)

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

#%%
# -----------------------
#
# Mean sea level pressure
#
# -----------------------
mslp_daily_averages = []

for index, time in enumerate(dt_array):
    mslp_dataset = isel_time_dataset(days_3_before_mslp, index)
    
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

#%%
# -----------------------
#
# Air Temperature 
#
# -----------------------
array = []

for index, time in enumerate(dt_array):
    level_airt_arr = []
    
    airt_dataset = isel_time_dataset(days_3_before_airt, index)
    lat = airt_dataset.latitude
    lon = airt_dataset.longitude
    airt_coords = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)
    airt_tree = cKDTree(airt_coords)
    haitang_coord = HAITANG_coords[index]
    
    indices = airt_tree.query_ball_point(haitang_coord, dis_thres)
    
    airt_200_points = airt_coords[indices]
    
    lat = airt_200_points[:,0]
    lon = airt_200_points[:,-1]
    new_airt_data = sel_lat_lon_dataset(airt_dataset, lat, lon)
    
    for level in range(len(airt_dataset.level.data)):
        dataset = isel_level_dataset(new_airt_data, level)
        airt_arr = dataset.t.data
        airt_mean = np.mean(airt_arr)
        level_airt_arr.append(airt_mean)
        
    array.append(level_airt_arr)

#%%
dt_array
level_arr


# %%
dims = dict(
    time = dt_array,
    level = level_arr
)
data_vars = {
    'airt': (['time', 'level'], array),
    'shum': (['time', 'level'], shum_mean_arr),
    'mslp': (['time'], mslp_daily_averages),
    'sst' : (['time'], sst_daily_averages)
}
#%%

dataset = xr.Dataset(data_vars, coords=dims)

# %%
nc_path = 'input_data.nc'
dataset.to_netcdf(path=nc_path)
dataset
# %%
dataset = xr.open_dataset('input_data.nc')
dataset
# %%

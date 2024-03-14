#%%
from MPI_ex.python_sang_ex.sang_ex.pyPI import pi
import MPI_ex.python_sang_ex.sang_ex.methods as mt 

import xarray as xr
import pickle
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import numba as nb

#%% Extracting Haitang from IBTrACS
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp'), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True) 
# Haitang 2005

wind_array = tc_HAITANG.usa_wind.data[0]
gen_index = mt.gen(wind_array)
lmi_index = mt.lmi(wind_array)
selected_indices = np.arange(gen_index, lmi_index+1)
haitang_dataset = tc_HAITANG.isel(date_time=selected_indices, drop=True)

#%%
haitang_dataset
#%%
# Substraction 3 days in  variable 'time' 
haitang_time = haitang_dataset['time'] - pd.Timedelta(days=3)

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
#%%
HAITANG_coords
#%%
#%% Air Temp, Specific Hum, MSLP, SST from ERA5, OISST 
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')
#%%
airt_dataset

#%% level array
level_arr = airt_dataset.level.data[::-1] # sorting downgrade 
level_arr
#%% dis threshold, fitering dataset with dt_array

dis_thres = 200/111

before_3_shum_dataset = shum_dataset.sel(time=dt_array, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=dt_array, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=dt_array, drop=True)
days_3_before_airt = airt_dataset.sel(time=dt_array, drop=True)

#%%
before_3_shum_dataset
#%%
# -------------------
#
# Specific Humidity
#
# -------------------
shum_mean_arr = []

for index, time in enumerate(dt_array):
    shum_arr = []
    
    shum_dataset = mt.isel_time_dataset(before_3_shum_dataset, index)
    
    shum_lat = shum_dataset.latitude.data
    shum_lon = shum_dataset.longitude.data
    
    new_lat, new_lon = mt.points_200(shum_lat, shum_lon, HAITANG_coords, index)
    
    len_level = len(shum_dataset.level.data)
    filtered_shum_dataset = mt.sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
    
    for level in range(len_level):
        dataset = mt.isel_level_dataset(filtered_shum_dataset, level)
        
        shum = dataset.q.data
        shum_mean = np.mean(shum)
        shum_arr.append(shum_mean)
        
    hum = shum_arr[::-1]
    shum_mean_arr.append(hum)
#%%
# -------------------------
#
# Sea Sureface Temperature
#
# -------------------------
sst_daily_averages = []

for index, time in enumerate(dt_array):
    oisst_dataset = mt.isel_time_dataset(days_3_before_oisst, index)

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
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# -----------------------
mslp_daily_averages = []

for index, time in enumerate(dt_array):
    mslp_dataset = mt.isel_time_dataset(days_3_before_mslp, index)
    
    mslp_lat = mslp_dataset.latitude
    mslp_lon = mslp_dataset.longitude
    mslp_value = (mslp_dataset.msl.data/100).flatten() # Pa to hPa
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
# t's unit is K, so change to degC ( - 273.15 )
#
# -----------------------
airt_array = []

for index, time in enumerate(dt_array):
    level_airt_arr = []
    
    airt_dataset = mt.isel_time_dataset(days_3_before_airt, index)
    lat = airt_dataset.latitude
    lon = airt_dataset.longitude
    
    new_lat, new_lon = mt.points_200(lat, lon, HAITANG_coords, index)
    len_level = len(airt_dataset.level.data)
    new_airt_data = mt.sel_lat_lon_dataset(airt_dataset, new_lat, new_lon)
    
    for level in range(len_level):
        dataset = mt.isel_level_dataset(new_airt_data, level)
        airt_arr = dataset.t.data - 273.15
        airt_mean = np.mean(airt_arr)
        level_airt_arr.append(airt_mean)
        
    airt = level_airt_arr[::-1]
    airt_array.append(airt)
# %%
dims = dict(
    time = dt_array,
    level = level_arr
)

lsm_arr = np.ones((len(dt_array), len(level_arr)))

data_vars = {
    'lsm': (['time', 'level'], lsm_arr),
    't': (['time', 'level'], airt_array),
    'q': (['time', 'level'], shum_mean_arr),
    'msl': (['time'], mslp_daily_averages),
    'sst' : (['time'], sst_daily_averages),
}

#%%
dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_data_preSST.nc'
dataset.to_netcdf(path=nc_path)
dt = xr.open_dataset(nc_path)

# %%
df = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_data_preSST.nc'
ds = mt.run_sample_dataset(df, CKCD=0.9)
ds.to_netcdf('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_output_preSST.nc')
preSST_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_output_preSST.nc')
preSST_dataset['vmax'] = preSST_dataset['vmax'] * 1.94384

#%%
preSST_dataset
# %%

durSST_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_output_durSST.nc')
durSST_dataset['vmax'] = durSST_dataset['vmax'] * 1.94384
#%%

# %%
pre_int = np.round(preSST_dataset.vmax.data)
dur_int = np.round(durSST_dataset.vmax.data)
#%%

for i in range(len(pre_int)):
    print(f'usa_wind : {haitang_dataset.usa_wind.data[0][i]} pre : {pre_int[i]} dur : {dur_int[i]}')

# %%
"""
usa_wind : 35.0 pre : 58.0 dur : 60.0
usa_wind : 35.0 pre : 60.0 dur : 64.0
usa_wind : 35.0 pre : 59.0 dur : 67.0
usa_wind : 35.0 pre : 60.0 dur : 66.0
usa_wind : 35.0 pre : 61.0 dur : 69.0
usa_wind : 35.0 pre : 59.0 dur : 71.0
usa_wind : 50.0 pre : 59.0 dur : 76.0
usa_wind : 65.0 pre : 63.0 dur : 83.0
usa_wind : 70.0 pre : 68.0 dur : 85.0
usa_wind : 75.0 pre : 83.0 dur : 92.0
usa_wind : 75.0 pre : 84.0 dur : 102.0
usa_wind : 85.0 pre : 84.0 dur : 111.0
usa_wind : 90.0 pre : 95.0 dur : 114.0
usa_wind : 90.0 pre : 107.0 dur : 120.0
usa_wind : 100.0 pre : 106.0 dur : 128.0
usa_wind : 115.0 pre : 111.0 dur : 131.0
usa_wind : 120.0 pre : 118.0 dur : 133.0
usa_wind : 130.0 pre : 137.0 dur : 135.0
usa_wind : 135.0 pre : 130.0 dur : 136.0
usa_wind : 140.0 pre : 126.0 dur : 135.0
"""
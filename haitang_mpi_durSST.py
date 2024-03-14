#%%
from MPI_ex.python_sang_ex.sang_ex.pyPI import pi
import MPI_ex.python_sang_ex.sang_ex.methods as mt 

import xarray as xr
import pickle
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import numba as nb
import statistics

#%% Extracting Haitang from IBTrACS
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp'), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True) # Haitang 2005

wind_array = tc_HAITANG.usa_wind.data[0]
gen_index = mt.gen(wind_array)
lmi_index = mt.lmi(wind_array)
selected_indices = np.arange(gen_index, lmi_index+1)
haitang_dataset = tc_HAITANG.isel(date_time=selected_indices, drop=True)
# %%
haitang_dataset
# %%
haitang_time = haitang_dataset['time'] - pd.Timedelta(days=3)
haitang_time
# %%
dt_array = []
for data in haitang_time[0]:
    datetime = np.array(data, dtype='datetime64[ns]')
    time = np.datetime64(datetime, 'D')
    dt_array.append(time)
# %%
dt_array
# %%
lat = np.array(haitang_dataset.usa_lat[0], dtype=object)
lon = np.array(haitang_dataset.usa_lon[0], dtype=object)
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lat,lon))
HAITANG_coords
# %%
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')


# %%
level_arr = airt_dataset.level.data[::-1] # sorting downgrade 
# %%
dis_thres = 200/111

before_3_shum_dataset = shum_dataset.sel(time=dt_array, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=dt_array, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=dt_array, drop=True)
days_3_before_airt = airt_dataset.sel(time=dt_array, drop=True)
#%%
days_3_before_airt
# %%
arr = []
arr_time_len = len(haitang_dataset.date_time)

for i in range(arr_time_len):
    haitang_i = haitang_dataset.isel(date_time=i, drop=True)
    
    data = {
        'time': dt_array[i],
        'usa_wind' : haitang_i.usa_wind.data[0],
        'usa_lat' : haitang_i.usa_lat.data[0],
        'usa_lon' : haitang_i.usa_lon.data[0],
        'usa_r34': haitang_i.usa_r34.data[0],
        'storm_speed': haitang_i.storm_speed.data[0]
    }
    
    arr.append(data)
    
df = pd.DataFrame(arr)
df.head(20)

all_r34_values = []
for r34_list in df['usa_r34']:
    # 각 리스트에 대해 NaN이 아닌 값만 추가
    all_r34_values.extend([val for val in r34_list if not np.isnan(val)])

# 모든 유효한 'usa_r34' 값의 평균 계산
average_r34 = np.mean(all_r34_values)
#%%

ds = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/THETA/2005/THETA.1440x720x50.20050710.nc')
ds 
# %%

tc_rad_dres = average_r34 / 111

# %% 
# ----------------------------
#            DAT
# ----------------------------
dat_list =[]

for index, row in df.iterrows():
    list_1 = []
    lat = row.usa_lat
    lon = row.usa_lon
    date = row.time
    
    salt_dataset, opt_dataset = mt.open_dataset(date)
        
    tc_coord = np.array(
        np.meshgrid(lon, lat)
    ).T.reshape(-1, 2)
    
    inter_sal, inter_opt, D = mt.interpolation(salt_dataset, opt_dataset, tc_coord[0], tc_rad_dres)
    TS = row.storm_speed * 0.514444    
    
    if np.all(np.isnan(row.usa_r34)):
        R = average_r34 * 1852
    else:
        R = np.nanmean(row.usa_r34) * 1852
    
    if np.isnan(R):
        R = 0
    else :
        R = R * 1852
        
    Vmax = row.usa_wind * 0.514444    
    
    for i in range(len(inter_opt.LATITUDE_T)):
        S = inter_sal.SALT.data[0][:, i, i].flatten()
        T = inter_opt.THETA.data[0][:, i, i].flatten()
        
        Dmix, Tmix = mt.mixing_depth(D, T, S, Vmax, TS, R)
        
        list_1.append(Tmix)           
    
    mean_1 = statistics.mean(list_1)
    print(f'{tc_coord} {mean_1}')
    
    dat_list.append(mean_1)
    
# %%
#%%
# -------------------
# Specific Humidity
# -------------------
shum_mean_arr = []

for index, time in enumerate(dt_array):
    shum_arr = []
    
    shum_dataset = mt.isel_time_dataset(before_3_shum_dataset, index)
    
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
    filtered_shum_dataset = mt.sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
    
    for level in range(len_level):
        dataset = mt.isel_level_dataset(filtered_shum_dataset, level)
        
        shum = dataset.q.data
        shum_mean = np.mean(shum)
        shum_arr.append(shum_mean)
        
        
    shum_mean_arr.append(shum_arr[::-1])

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
    mslp_value = (mslp_dataset.msl.data/100).flatten()
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
# Air Temperature 
# t's unit is K, so change to degC ( - 273.15 )
# -----------------------
array = []

for index, time in enumerate(dt_array):
    level_airt_arr = []
    
    airt_dataset = mt.isel_time_dataset(days_3_before_airt, index)
    lat = airt_dataset.latitude
    lon = airt_dataset.longitude
    airt_coords = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)
    airt_tree = cKDTree(airt_coords)
    haitang_coord = HAITANG_coords[index]
    
    indices = airt_tree.query_ball_point(haitang_coord, dis_thres)
    
    airt_200_points = airt_coords[indices]
    
    lat = airt_200_points[:,0]
    lon = airt_200_points[:,-1]
    new_airt_data = mt.sel_lat_lon_dataset(airt_dataset, lat, lon)
    
    for level in range(len(airt_dataset.level.data)):
        dataset = mt.isel_level_dataset(new_airt_data, level)
        airt_arr = dataset.t.data - 273.15
        airt_mean = np.mean(airt_arr)
        level_airt_arr.append(airt_mean)
        
    array.append(level_airt_arr[::-1])


# %%
dims = dict(
    time = dt_array,
    level = level_arr
)


# %%
lsm_arr = np.ones((len(dt_array), len(level_arr)))

data_vars = {
    'lsm': (['time', 'level'], lsm_arr),
    't': (['time', 'level'], array),
    'q': (['time', 'level'], shum_mean_arr),
    'msl': (['time'], mslp_daily_averages),
    'sst' : (['time'], dat_list),
}

dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_data_durSST.nc'
dataset.to_netcdf(path=nc_path)
#%%
dt = xr.open_dataset(nc_path)
#%%
dt

#%%
df = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_data_durSST.nc'
ds = mt.run_sample_dataset(df, CKCD=0.9)
ds.to_netcdf('0307_output_durSST.nc')

# %%
durSST_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_output_durSST.nc')
durSST_dataset['vmax'] = durSST_dataset['vmax'] * 1.94384
#%%

np.round(durSST_dataset.vmax.data)
"""
[ 60.,  64.,  67.,  66.,  69.,  71.,  76.,  83.,  85.,  92., 102.,
       111., 114., 120., 128., 131., 133., 135., 136., 135.]

"""
# %%
haitang_dataset.usa_wind.data[0]
"""
[ 35.,  35.,  35.,  35.,  35.,  35.,  50.,  65.,  70.,  75.,  75.,
        85.,  90.,  90., 100., 115., 120., 130., 135., 140.]
"""
# %%

#%%
import MPI_ex.python_sang_ex.sang_ex.methods as mt 

import xarray as xr
import numpy as np
import pandas as pd
import statistics

#%% Extracting Haitang from IBTrACS
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
haitang_dataset = mt.preprocess_IBTrACS(dataset, b'2005192N22155', b'jtwc_wp')
#%%
haitang_dataset
# %%
haitang_pre_time = haitang_dataset['time'] - pd.Timedelta(days=3)
haitang_time = haitang_dataset['time']
# %%
pre_dt = []
for data in haitang_pre_time[0]:
    datetime = np.array(data, dtype='datetime64[ns]')
    time = np.datetime64(datetime, 'D')
    pre_dt.append(time)

dt = []    
for data in haitang_time[0]:
    datetime = np.array(data, dtype='datetime64[ns]')
    time = np.datetime64(datetime, 'D')
    dt.append(time)
    
# %%
lat = np.array(haitang_dataset.usa_lat[0], dtype=object)
lon = np.array(haitang_dataset.usa_lon[0], dtype=object)
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lon,lat))
# %%
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')

#%%
airt_dataset.t.data[0][11]
# %%
level_arr = airt_dataset.level.data[::-1] # sorting downgrade 
# %%
before_3_shum_dataset = shum_dataset.sel(time=dt, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=pre_dt, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=pre_dt, drop=True)
days_3_before_airt = airt_dataset.sel(time=dt, drop=True)

# %%
arr = []
arr_time_len = len(haitang_dataset.date_time)

for i in range(arr_time_len):
    haitang_i = haitang_dataset.isel(date_time=i, drop=True)
    
    data = {
        'time': pre_dt[i],
        'usa_wind' : haitang_i.usa_wind.data[0],
        'usa_lat' : haitang_i.usa_lat.data[0],
        'usa_lon' : haitang_i.usa_lon.data[0],
        'usa_r34': haitang_i.usa_r34.data[0],
        'storm_speed': haitang_i.storm_speed.data[0],
        'usa_rmw' : haitang_i.usa_rmw.data[0]
    }
    
    arr.append(data)
    
df = pd.DataFrame(arr)
df.head(20)

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
    tc_rad = row.usa_rmw
    salt_dataset, opt_dataset = mt.open_dataset(date)
    
    tc_coord = [lon, lat]
    
    new_sal_lon, new_sal_lat = mt.points_200(salt_dataset, tc_coord)
    new_opt_lon, new_opt_lat = mt.points_200(opt_dataset, tc_coord)
    
    new_sal_dataset = mt.sel_opt_sal_dataset(salt_dataset, new_sal_lat, new_sal_lon)
    new_opt_dataset = mt.sel_opt_sal_dataset(opt_dataset,new_opt_lat, new_opt_lon)

    depth = new_sal_dataset.DEPTH_T.data
    len_lat = len(new_sal_dataset.LATITUDE_T)
    sal = new_sal_dataset.SALT.data[0]
    opt = new_opt_dataset.THETA.data[0]
    
    D = depth
    TS = row.storm_speed * 0.5144
    R = tc_rad * 1609.34
    Vmax = row.usa_wind * 0.5144
    
    for i in range(len_lat):
        S = sal[:, i, i].flatten()
        T = opt[:, i, i].flatten()     
        
        Dmix, Tmix = mt.mixing_depth(D, T, S, Vmax, TS, R)
               
        list_1.append(Tmix)           
        
        print(f'{Dmix} {list_1}')
        mean_1 = statistics.mean(list_1)
        
    dat_list.append(mean_1)
    print('=========================================')
  

#%%
# -------------------
# Specific Humidity
# -------------------
shum_mean_arr = []
haitang_rmw = haitang_dataset.usa_rmw.data[0]

for index, time in enumerate(dt):
    shum_arr = []
    haitang_coord = HAITANG_coords[index]
    
    shum_dataset = mt.isel_time_dataset(before_3_shum_dataset, index)
    new_lon, new_lat = mt.points_200(shum_dataset, haitang_coord)
    
    len_level = len(shum_dataset.level.data)
    filtered_shum_dataset = mt.sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
    
    for level in range(len_level):
        dataset = mt.isel_level_dataset(filtered_shum_dataset, level)
        
        shum = dataset.q.data * 1000
        
        shum_mean = np.mean(shum)
        print(shum_mean)
        shum_arr.append(shum_mean)
        print('-============================================')
    
    print(shum_arr)
    shum_mean_arr.append(shum_arr[::-1])
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')

print(shum_mean_arr)
#%%
# -----------------------
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# -----------------------
mslp_daily_averages = []

for index, time in enumerate(pre_dt):
    pressure_values = []
    mslp_dataset = mt.isel_time_dataset(days_3_before_mslp, index)
    haitang_index_coord = HAITANG_coords[index]
    
    new_lon, new_lat = mt.points_200(mslp_dataset, haitang_index_coord)
    filtered_mslp_dataset = mt.sel_lat_lon_dataset(mslp_dataset, new_lat, new_lon)
    
    msl_arr = filtered_mslp_dataset.msl.data[0]/100
    print(f'{time}  {msl_arr}')
    msl_mean = np.mean(msl_arr)
    mslp_daily_averages.append(msl_mean)
    
mslp_daily_averages
#%%
# -----------------------
# Air Temperature 
# t's unit is K, so change to degC ( - 273.15 )
# -----------------------
airt_array = []
haitang_rmw = haitang_dataset.usa_rmw.data[0]

for index, time in enumerate(dt):
    level_airt_arr = []
    
    airt_dataset = mt.isel_time_dataset(days_3_before_airt, index)
    haitang_coord = HAITANG_coords[index]
    
    new_lon, new_lat = mt.points_200(airt_dataset, haitang_coord)
    new_airt_data = mt.sel_lat_lon_dataset(airt_dataset, new_lat, new_lon)
    
    for level in range(len(airt_dataset.level.data)):
        dataset = mt.isel_level_dataset(new_airt_data, level)
        airt_arr = dataset.t.data - 273.15
        print(f'{time} {level} {airt_arr}')
        airt_mean = np.mean(airt_arr)
        
        level_airt_arr.append(airt_mean)
        print('==================================')
        
    
    airt_array.append(level_airt_arr[::-1])
    print('++++++++++++++++++++++++++++++++++++++++++')    

print(airt_array)
# %%
dims = dict(
    time = pre_dt,
    level = level_arr
)


# %%
lsm_arr = np.ones((len(pre_dt), len(level_arr)))

data_vars = {
    'lsm': (['time', 'level'], lsm_arr),
    't': (['time', 'level'], airt_array),
    'q': (['time', 'level'], shum_mean_arr),
    'msl': (['time'], mslp_daily_averages),
    'sst' : (['time'], dat_list),
}

dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_data_durSST.nc'
dataset.to_netcdf(path=nc_path)
#%%

dt = xr.open_dataset(nc_path)
dt

#%%
df = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_data_durSST.nc'
ds = mt.run_sample_dataset(df, CKCD=0.9)
ds.to_netcdf('0307_output_durSST.nc')

#%%
ds
# %%
durSST_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_output_durSST.nc')
#%%
durSST_dataset['vmax'] = durSST_dataset['vmax'] * 1.94384 # m/s to knots 
durSST_dataset
#%%

np.round(durSST_dataset.vmax.data)

# %%
durSST_dataset

# %%
ex = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_output_preSST.nc')
# %%
ex
# %%

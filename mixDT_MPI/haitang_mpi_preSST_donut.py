#%%
import MPI_ex.python_sang_ex.sang_ex.mixingdt_mpi.methods as mt 

import xarray as xr
import numpy as np
import pandas as pd

#%% Extracting Haitang from IBTrACS
haitang_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/haitang_2005.nc')
#%%
# Substraction 3 days in  variable 'time' 
haitang_pre_time = haitang_dataset['time'] - pd.Timedelta(days=3)
haitang_time = haitang_dataset['time']

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
    
#%%
pre_dt
#%%
dt
#%%
lat = np.array(haitang_dataset.usa_lat[0], dtype=object)
lon = np.array(haitang_dataset.usa_lon[0], dtype=object)
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lon,lat))
#%% Air Temp, Specific Hum, MSLP, SST from ERA5, OISST 
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')
#%%
oisst_DATA
#%% level array
level_arr = airt_dataset.level.data[::-1] # sorting downgrade 
level_arr
#%% dis threshold, fitering dataset with dt_array

before_3_shum_dataset = shum_dataset.sel(time=dt, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=pre_dt, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=pre_dt, drop=True)
days_3_before_airt = airt_dataset.sel(time=dt, drop=True)

#%%
days_3_before_mslp
#%%
# -------------------
#
# Specific Humidity
# kg kg**-1(kg/kg) to g/kg by multipling 1000
#
# -------------------
shum_mean_arr = []
haitang_rmw = haitang_dataset.usa_rmw.data[0]

for index, time in enumerate(dt):
    shum_arr = []
    haitang_coord = HAITANG_coords[index]
    tc_thres = (haitang_rmw[index]*1.609) / 111
    
    shum_dataset = mt.isel_time_dataset(before_3_shum_dataset, index)
    new_lon, new_lat = mt.donut_points(shum_dataset, haitang_coord, tc_thres)
    
    len_level = len(shum_dataset.level.data)
    filtered_shum_dataset = mt.sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
    
    for level in range(len_level):
        dataset = mt.isel_level_dataset(filtered_shum_dataset, level)
        
        shum = dataset.q.data * 1000
        
        shum_mean = np.mean(shum)
        shum_arr.append(shum_mean)
    
    shum_mean_arr.append(shum_arr[::-1])
#%%
shum_mean_arr
#%%
# -------------------------
#
# Sea Sureface Temperature
#
# -------------------------
sst_daily_averages = []
haitang_rmw = haitang_dataset.usa_rmw.data[0]

for index, time in enumerate(dt):
    oisst_dataset = mt.isel_time_dataset(days_3_before_oisst, index)

    haitang_index_coord = HAITANG_coords[index]
    tc_thres = (haitang_rmw[index]*1.609) / 111
    
    new_lon, new_lat = mt.donut_points(oisst_dataset, haitang_index_coord, tc_thres)
    filtered_sst_dataset = mt.sel_lat_lon_dataset(oisst_dataset, new_lat, new_lon)
    
    sst = filtered_sst_dataset.sst.data[0]
    sst_average = np.mean(sst) 
    sst_daily_averages.append(sst_average)

print(sst_daily_averages)
#%%
# -----------------------
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# -----------------------
mslp_daily_averages = []
haitang_rmw = haitang_dataset.usa_rmw.data[0]

for index, time in enumerate(pre_dt):
    pressure_values = []
    mslp_dataset = mt.isel_time_dataset(days_3_before_mslp, index)
    tc_thres = (haitang_rmw[index]*1.609) / 111
    haitang_index_coord = HAITANG_coords[index]
    
    new_lon, new_lat = mt.donut_points(mslp_dataset, haitang_index_coord, tc_thres)
    filtered_mslp_dataset = mt.sel_lat_lon_dataset(mslp_dataset, new_lat, new_lon)
    
    msl_arr = filtered_mslp_dataset.msl.data[0]/100
    msl_mean = np.mean(msl_arr)
    mslp_daily_averages.append(msl_mean)
#%%
mslp_daily_averages

#%%
# -----------------------
#
# Air Temperature 
#
# t's unit is K, so change to degC ( - 273.15 )
#
# -----------------------
airt_array = []
haitang_rmw = haitang_dataset.usa_rmw.data[0]

for index, time in enumerate(dt):
    level_airt_arr = []
    
    airt_dataset = mt.isel_time_dataset(days_3_before_airt, index)
    tc_thres = (haitang_rmw[index]*1.609) / 111
    haitang_coord = HAITANG_coords[index]
    
    new_lon, new_lat = mt.donut_points(airt_dataset, haitang_coord, tc_thres)
    new_airt_data = mt.sel_lat_lon_dataset(airt_dataset, new_lat, new_lon)
    
    for level in range(len(airt_dataset.level.data)):
        dataset = mt.isel_level_dataset(new_airt_data, level)
        airt_arr = dataset.t.data - 273.15
        airt_mean = np.mean(airt_arr)
        level_airt_arr.append(airt_mean)
        
    airt_array.append(level_airt_arr[::-1])

print(airt_array)
# %%
dims = dict(
    time = pre_dt,
    level = level_arr
)

lsm_arr = np.ones((len(pre_dt), len(level_arr)))

data_vars = {
    'lsm': (['time', 'level'], lsm_arr),
    't': (['time', 'level'], airt_array),
    'q': (['time', 'level'], shum_mean_arr),
    'msl': (['time'], mslp_daily_averages),
    'sst' : (['time'], sst_daily_averages),
}

#%%
dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_data_preSST_donut.nc'
dataset.to_netcdf(path=nc_path)
dt = xr.open_dataset(nc_path)

# %%

df = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_data_preSST_donut.nc'
ds = mt.run_sample_dataset(df, CKCD=0.9)
ds.to_netcdf('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_output_preSST_donut.nc')

#%%

preSST_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/0307_output_preSST_donut.nc')
preSST_dataset['vmax'] = preSST_dataset['vmax'] * 1.94384 # m/s t Knots

#%%
mpi = preSST_dataset.vmax.data
sst = preSST_dataset.sst.data
time = preSST_dataset.time.data
msl = preSST_dataset.msl.data
airt = preSST_dataset.t.data
hum = preSST_dataset.q.data
#%%
for i in range(len(mpi)):
    time_formatted = pd.to_datetime(time[i]).strftime('%Y-%m-%d')    
    value = np.round(sst[i], 1)
    print(f'{value}')    


# %%
preSST_dataset
# %%

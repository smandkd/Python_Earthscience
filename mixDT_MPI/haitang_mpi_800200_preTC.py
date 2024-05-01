#%%
import mixing_dt_mpi.methods as mt 
import mixing_dt_mpi.mpi as mpi 

import xarray as xr
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')


#%% 
haitang_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/haitang_2005.nc')

haitang_date = haitang_dataset['time']
pre_dt = mt.TC_pre_3_date(haitang_date)
dt = mt.TC_present_date(haitang_date)
    
HAITANG_coords = mt.TC_usa_center_points(haitang_dataset)
#%% Air Temp, Specific Hum, MSLP, SST from ERA5, OISST 
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')
level_arr = mt.sort_level_downgrade(airt_dataset)
# %%
preset_shum_dataset = shum_dataset.sel(time=dt, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=pre_dt, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=pre_dt, drop=True)
present_airt = airt_dataset.sel(time=dt, drop=True)
#%%
# -------------------
#
# Specific Humidity
# kg kg**-1(kg/kg) to g/kg by multipling 1000
#
# -------------------
shum_mean_arr = []
shum_10_list = []
shum_mean_arr, shum_10_list = mt.shum_donut_mean(HAITANG_coords, preset_shum_dataset)

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
# ---------------------------------------
#
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
#
# ---------------------------------------
mslp_daily_averages = []

mslp_daily_averages = mt.mslp_donut_mean(pre_dt, HAITANG_coords, days_3_before_mslp)

#%%
# -----------------------
#
# Air Temperature 
#
# t's unit is K, so change to degC ( - 273.15 )
#
# -----------------------
airt_array = []
airt_10_list = []
airt_array, airt_10_list = mt.airt_donut_mean(dt, HAITANG_coords, present_airt)

# %%
dims = dict(
    time = pre_dt,
    level = level_arr
)

lsm_arr = np.ones((len(pre_dt), len(level_arr)))

data_vars = {
    'sst' : (['time'], sst_daily_averages),
    'lsm': (['time', 'level'], lsm_arr),
    'airt': (['time', 'level'], airt_array),
    'shum': (['time', 'level'], shum_mean_arr),
    'mslp': (['time'], mslp_daily_averages),
    'shum_10m': (['time'], shum_10_list),
    'airt_10m': (['time'], airt_10_list),
}

#%%
dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_haitang_input_preSST_MPI_donut.nc'
dataset.to_netcdf(path=nc_path)
input_ds = xr.open_dataset(nc_path)
#%%

input_ds
# %%
output_ds = mpi.run_sample_dataset(nc_path, CKCD=0.9)
output_ds.to_netcdf('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_haitang_output_preSST_MPI_donut.nc')

#%%

preSST_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_output_preSST_MPI_donut.nc')
preSST_dataset['vmax'] = preSST_dataset['vmax'] * 1.94384 # m/s t Knots

#%%
vmax = preSST_dataset.vmax.data
sst = preSST_dataset.sst.data
time = preSST_dataset.time.data
msl = preSST_dataset.mslp.data
airt = preSST_dataset.airt.data
hum = preSST_dataset.shum.data
#%%
for i in range(len(vmax)):
    time_formatted = pd.to_datetime(time[i]).strftime('%Y-%m-%d')    
    value = np.round(sst[i], 1)
    print(f'{value}')    

# %%

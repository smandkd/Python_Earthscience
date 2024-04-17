#%%
import MPI_ex.python_sang_ex.sang_ex.methods as mt 

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import cartopy.crs as crs
import cartopy.feature as cfeature
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

time_formatted = pd.to_datetime(pre_dt).strftime('%Y-%m-%d')
pre_arr = np.array(time_formatted)
#%%
len(pre_dt)
time_arr = np.arange(1, 21)

#%%
dt
#%%
lat = np.array(np.round(haitang_dataset.usa_lat[0],2))
lon = np.array(np.round(haitang_dataset.usa_lon[0],2))
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lon, lat))
#%%
HAITANG_coords
#%% Air Temp, Specific Hum, MSLP, SST from ERA5, OISST 
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')

#%% level array
level_arr = airt_dataset.level.data[::-1] # sorting downgrade 
level_arr
#%% dis threshold, fitering dataset with dt_array

before_3_shum_dataset = shum_dataset.sel(time=dt, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=pre_dt, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=pre_dt, drop=True)
days_3_before_airt = airt_dataset.sel(time=dt, drop=True)

#%%
# -------------------
#
# Specific Humidity
# kg kg**-1(kg/kg) to g/kg by multipling 1000
#
# -------------------
shum_mean_arr = []
haitang_rmw = haitang_dataset.usa_rmw.data[0]
donut_lon = []
donut_lat = []

for index, time in enumerate(dt):
    shum_arr = []
    haitang_coord = HAITANG_coords[index]
    tc_thres = 200 / 111
    
    shum_dataset = mt.isel_time_dataset(before_3_shum_dataset, index)
    new_lon, new_lat = mt.donut_points(shum_dataset, haitang_coord, tc_thres)
    
    donut_lon.append(new_lon)
    donut_lat.append(new_lat)
    
    len_level = len(shum_dataset.level.data)
    filtered_shum_dataset = mt.sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
    
    for level in range(len_level):
        dataset = mt.isel_level_dataset(filtered_shum_dataset, level)
        
        shum = dataset.q.data * 1000
        
        print(shum)
        shum_mean = np.nanmean(shum)
        shum_arr.append(shum_mean)
    
    shum_mean_arr.append(shum_arr[::-1]) 
#%%
lat = donut_lat[1]
lon = donut_lon[1]

#%%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.set_global()
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.gridlines()
ax.plot(lon, lat, transform=crs.PlateCarree(), linewidth=1)
plt.savefig("haitang_800200.pdf")
plt.show()

#%%
shum_mean_arr
#%%
# -------------------------
#
# Sea Sureface Temperature
#
# -------------------------
sst_daily_averages = []

for index, time in enumerate(dt):
    oisst_dataset = mt.isel_time_dataset(days_3_before_oisst, index)

    haitang_index_coord = HAITANG_coords[index]
    tc_thres = 200 / 111
    
    new_lon, new_lat = mt.donut_points(oisst_dataset, haitang_index_coord, tc_thres)
    print(f'donut lon, lat : {new_lon} {new_lat}')
    filtered_sst_dataset = mt.sel_lat_lon_dataset(oisst_dataset, new_lat, new_lon)
    
    sst = filtered_sst_dataset.sst.data[0]
    print(sst)
    sst_average = np.nanmean(sst) 
    sst_daily_averages.append(sst_average)
    
   
#%%
print(sst_daily_averages)
#%%
plt.figure(figsize=(10, 6))
plt.plot(time_arr, sst_daily_averages, 'bo-')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('sst')
plt.show()
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
    tc_thres = 200 / 111
    haitang_index_coord = HAITANG_coords[index]
    
    new_lon, new_lat = mt.donut_points(mslp_dataset, haitang_index_coord, tc_thres)
    filtered_mslp_dataset = mt.sel_lat_lon_dataset(mslp_dataset, new_lat, new_lon)
    
    msl_arr = filtered_mslp_dataset.msl.data[0]/100
    msl_mean = np.nanmean(msl_arr)
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
    tc_thres = 200 / 111
    haitang_coord = HAITANG_coords[index]
    
    new_lon, new_lat = mt.donut_points(airt_dataset, haitang_coord, tc_thres)
    new_airt_data = mt.sel_lat_lon_dataset(airt_dataset, new_lat, new_lon)
    
    for level in range(len(airt_dataset.level.data)):
        dataset = mt.isel_level_dataset(new_airt_data, level)
        airt_arr = dataset.t.data - 273.15
        airt_mean = np.nanmean(airt_arr)
        level_airt_arr.append(airt_mean)
        
    airt_array.append(level_airt_arr[::-1])
#%%
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
nc_path = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_input_preSST_donut.nc'
dataset.to_netcdf(path=nc_path)
dt = xr.open_dataset(nc_path)

# %%
ds = mt.run_sample_dataset(nc_path, CKCD=0.9)
ds.to_netcdf('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_output_preSST_donut.nc')

#%%

preSST_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_output_preSST_donut.nc')
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
    value = np.mean(np.round(mpi[i], 1))
    print(f'{value}')    


# %%
plt.figure(figsize=(10, 6))
plt.plot(time_arr, sst, 'bo-')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('SST')
plt.show()
# %%
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('mpi')
ax1.plot(time_arr, mpi, 'bo-', color='green', label='mpi')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.set_ylabel('sst')
ax2.plot(time_arr, sst, 'bo-', color='deeppink', label='sst')
ax2.legend(loc='lower right')

plt.xticks(np.arange(1, 21, step=1))
plt.show()
# %%
hum
mean_arr_hum = []
for h in hum:
    mean = np.mean(h)
    print(mean)
    mean_arr_hum.append(mean)

mean_arr_airt = []
for t in airt:
    mean = np.mean(t)
    print(mean)
    mean_arr_airt.append(mean)
    
plt.subplot(2, 2, 1)
plt.plot(time_arr, mpi, 'bo-', color='green')
plt.title('PRE_MPI')
plt.ylabel('mpi')

plt.subplot(2, 2, 2)
plt.plot(time_arr, sst, 'bo-', color='red')
plt.title('SST')
plt.ylabel('sst')

plt.subplot(2, 2, 3)
plt.plot(time_arr, mean_arr_hum, 'bo-', color='deeppink')
plt.title('Humidity')
plt.ylabel('hum')

plt.subplot(2, 2, 4)
plt.plot(time_arr, mean_arr_airt, 'bo-', color='blue')
plt.title('AIRT')
plt.ylabel('airt')
# %%

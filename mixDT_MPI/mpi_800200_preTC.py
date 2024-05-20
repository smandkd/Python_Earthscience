
#%%
import MPI_ex.MPI_mixing_depth_temp.mixing_dt_mpi.methods as mt 
import MPI_ex.MPI_mixing_depth_temp.mixing_dt_mpi.mpi as mpi 

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import warnings
warnings.filterwarnings(action='ignore')
#%%
tc_name = 'haitang'
tc_sid = b'2005192N22155'
tc_agency = b'jtwc_wp'

#%% 
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
tc_dataset = mt.preprocess_IBTrACS(dataset, tc_sid, tc_agency)
tc_dataset
#%%
tc_date = tc_dataset['time']

pre_dt = mt.TC_pre_3_date(tc_dataset)
dt = mt.TC_present_date(tc_dataset)

formatted_dates = [f"{date.astype(object).month}/{date.astype(object).day}" for date in dt]
formatted_dates

tc_coords = mt.TC_usa_center_points(tc_dataset)
#%% Air Temp, Specific Hum, MSLP, SST from ERA5, OISST 
years_months = sorted(set([(d.astype('datetime64[Y]').astype(int) + 1970, d.astype('datetime64[M]').astype(int) % 12 + 1) for d in dt]))
year = years_months[0][0]
month = years_months[0][1]
#%%
airt_path = f'/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_{year:04d}{month:02d}.nc'
shum_path = f'/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_{year:04d}{month:02d}.nc'
mslp_path = f'/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_{year:04d}{month:02d}.nc'
sst_path = f'/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.{year:04d}.nc'
airt_dataset = xr.open_dataset(airt_path)
shum_dataset = xr.open_dataset(shum_path)
mslp_dataset = xr.open_dataset(mslp_path)
sst_dataset = xr.open_dataset(sst_path)

level_arr = mt.sort_level_downgrade(airt_dataset)
# %%
preset_shum_dataset = shum_dataset.sel(time=dt, drop=True)
days_3_before_oisst = sst_dataset.sel(time=pre_dt, drop=True)
days_3_before_mslp = mslp_dataset.sel(time=pre_dt, drop=True)
present_airt = airt_dataset.sel(time=dt, drop=True)

df_tc = mt.create_TC_dataframe(tc_dataset)
#%%
# -------------------
#
# Specific Humidity
# kg kg**-1(kg/kg) to g/kg by multipling 1000
#
# -------------------
shum_arr = []
shum_10_list = []
shum_arr, shum_10_list = mt.shum_donut_mean(tc_coords, preset_shum_dataset)

#%%
# -------------------------
#
# Sea Sureface Temperature
#
# -------------------------
sst_arr = []
haitang_rmw = tc_dataset.usa_rmw.data[0]
donut_lat = []
donut_lon = []

for index, time in enumerate(dt):
    oisst_dataset = mt.isel_time_dataset(days_3_before_oisst, index)

    haitang_index_coord = tc_coords[index]
    # tc_thres = (haitang_rmw[index]*1.609) / 111
    tc_thres = 200 / 111
    
    new_lon, new_lat = mt.donut_points(oisst_dataset, haitang_index_coord, tc_thres)
    donut_lat.append(new_lat)
    donut_lon.append(new_lon)
    filtered_sst_dataset = mt.sel_lat_lon_dataset(oisst_dataset, new_lat, new_lon)
    
    sst = filtered_sst_dataset.sst.data[0]
    sst_average = np.nanmean(sst) 
    sst_arr.append(sst_average)
#%%
# ===========================================
#  Plot input area on map
# ===========================================
# flattened_longitude = np.concatenate(donut_lon)
# flattened_latitude = np.concatenate(donut_lat)
flattened_longitude = donut_lon[-1]
flattened_latitude = donut_lat[-1]

# Create a plot with Cartopy projection
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([120, 160, 10, 40], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgreen')
ax.add_feature(cfeature.OCEAN, facecolor='aqua')
ax.add_feature(cfeature.LAKES, facecolor='aqua')
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True)

# Plot points on the map
ax.scatter(flattened_longitude, flattened_latitude, s=10, color='red', marker='o', alpha=0.7, transform=ccrs.PlateCarree())

# Add title
plt.title('TC Points on Map')
plt.show()
#%%
# ---------------------------------------
#
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
#
# ---------------------------------------
mslp_arr = []

mslp_arr = mt.mslp_donut_mean(pre_dt, tc_coords, days_3_before_mslp)

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
airt_array, airt_10_list = mt.airt_donut_mean(dt, tc_coords, present_airt)

# %%
dims = dict(
    time = pre_dt,
    level = level_arr
)

lsm_arr = np.ones((len(pre_dt), len(level_arr)))

data_vars = {
    'sst' : (['time'], sst_arr),
    'lsm': (['time', 'level'], lsm_arr),
    'airt': (['time', 'level'], airt_array),
    'shum': (['time', 'level'], shum_arr),
    'mslp': (['time'], mslp_arr),
    'shum_10m': (['time'], shum_10_list),
    'airt_10m': (['time'], airt_10_list),
}

#%%
dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '/home/tkdals/homework_3/MPI_ex/data/800200_haitang_input_preSST_MPI_donut.nc'
dataset.to_netcdf(path=nc_path)
input_ds = xr.open_dataset(nc_path)

#%%
nc_path = '/home/tkdals/homework_3/MPI_ex/data/800200_'+ tc_name+'_input_preSST_MPI_donut.nc'
input_ds = xr.open_dataset(nc_path)
# %%
output_ds = mpi.run_sample_dataset(input_ds, CKCD=0.9)
#%%
output_ds.to_netcdf('/home/tkdals/homework_3/MPI_ex/data/800200_'+ tc_name+'_output_preSST_MPI_donut.nc')

#%%

preSST_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/data/800200_'+ tc_name +'_output_preSST_MPI_donut.nc')
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
    value = np.round(vmax[i], 1)
    print(f'{value}')    

#%%
# =============================================
# Plot MPI, TC wind speed
# =============================================


usa_wind = df_tc.usa_wind # unit : Knots
time_arr = np.arange(len(dt))
plt.figure(figsize=(10, 6))
plt.plot(time_arr, usa_wind, 'bo-', color='deeppink', label='usa_wind')
plt.plot(time_arr, vmax, 'bo-', color='purple', label='MPI')
plt.ylabel('wind speed(Knots)')
plt.title(tc_name + 'MPI, usa_wind')
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)  # 회전 추가로 레이블이 겹치지 않게 함
plt.yticks(fontsize=9)
plt.legend(loc='upper left')
plt.show()

#%%
# =============================================
# Plot MPI, Mixing temperature
# =============================================
time_arr = np.arange(0, len(dt))
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots()
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)
ax1.set_ylabel('mpi(knots)')
ax1.plot(time_arr, vmax, 'bo-', color='red', label='mpi')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.set_ylabel('mixing temperature(degreeC)')
ax2.plot(time_arr, sst, 'bo-', color='blue', label='mixing temp')
ax2.legend(loc='upper left')

# plt.title('Old mixing temperature, MPI 6 hours interval')
plt.title(tc_name + ' Mixing temperature, MPI')
plt.show()
# %%

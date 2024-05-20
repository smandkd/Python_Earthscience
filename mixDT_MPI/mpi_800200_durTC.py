#%%
import MPI_ex.MPI_mixing_depth_temp.mixing_dt_mpi.methods as mt 
import MPI_ex.MPI_mixing_depth_temp.mixing_dt_mpi.mpi as mpi 

import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import warnings
warnings.filterwarnings(action='ignore')
#%%
tc_name = 'haitang'
tc_sid = b'2005192N22155'
tc_agency = b'jtwc_wp'

# %%
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
tc_dataset = mt.preprocess_IBTrACS(dataset, tc_sid, tc_agency)
#%%
tc_dataset
#%%
tc_date = tc_dataset['time']

pre_dt = mt.TC_pre_3_date(tc_dataset)
dt = mt.TC_present_date(tc_dataset)

formatted_dates = [f"{date.astype(object).month}/{date.astype(object).day}" for date in dt]
formatted_dates

tc_coords = mt.TC_usa_center_points(tc_dataset)

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
#%%
present_shum_dataset = shum_dataset.sel(time=dt, drop=True)
days_3_before_oisst = sst_dataset.sel(time=pre_dt, drop=True)
days_3_before_mslp = mslp_dataset.sel(time=pre_dt, drop=True)
present_airt_dataset = airt_dataset.sel(time=dt, drop=True)
#%%
df_tc = mt.create_TC_dataframe(tc_dataset)
#%%
# ----------------------------
#            DAT
# ----------------------------

depth_list = [] # Dmix list

ecco2_saomai_lon = [] # longitude in TC radius
ecco2_saomai_lat = [] # latitude in TC radius
Tmix_list = []
Dmix_list = [] 
Theta_list = []
Salt_list = []
Dens_list = []
FT_list = []
Tx_list = []

Tmix_list, Dmix_list, ecco2_haitang_lon, ecco2_haitang_lat = mt.calculate_depth_mixing_d_t(df_tc)

print(f'Tmix : {np.round(Tmix_list, 2)}')
print(f'Dmix : {np.round(Dmix_list, 2)}')

#%%
flattened_longitude = np.concatenate(ecco2_haitang_lon)
flattened_latitude = np.concatenate(ecco2_haitang_lat)
# flattened_longitude = ecco2_haitang_lon[-1]
# flattened_latitude = ecco2_haitang_lat[-1]
# Create a plot with Cartopy projection
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([130, 180, 0, 40], crs=ccrs.PlateCarree())

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
plt.title('Longitude and Latitude Points on Map')
plt.show()
#%%
# -------------------
# Specific Humidity
# -------------------

shum_mean_arr = []
shum_10_list = []
shum_mean_arr, shum_10_list = mt.shum_donut_mean(tc_coords, present_shum_dataset)


#%%
# -----------------------
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# -----------------------
mslp_daily_averages = []

mslp_daily_averages = mt.mslp_donut_mean(pre_dt, tc_coords, days_3_before_mslp)

#%%
# -----------------------
# Air Temperature 
# t's unit is K, so change to degC ( - 273.15 )
# -----------------------
airt_array = []
airt_10_list = []
airt_array, airt_10_list = mt.airt_donut_mean(dt, tc_coords, present_airt_dataset)
# %%
dims = dict(
    time = dt,
    level = level_arr
)

lsm_arr = np.ones((len(pre_dt), len(level_arr)))

data_vars = {
    'lsm': (['time', 'level'], lsm_arr),
    'airt': (['time', 'level'], airt_array),
    'shum': (['time', 'level'], shum_mean_arr),
    'mslp': (['time'], mslp_daily_averages),
    'sst' : (['time'], Tmix_list),
    'mixD' : (['time'], Dmix_list),
    'shum_10m': (['time'], shum_10_list),
    'airt_10m': (['time'], airt_10_list),
}

dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '/home/tkdals/homework_3/MPI_ex/data/800200_' + tc_name + '_input_durSST_donut_ori.nc'
dataset.to_netcdf(path=nc_path)
#%%
input_ds = xr.open_dataset(nc_path)
input_ds
#%%
output_ds = mpi.run_sample_dataset(input_ds, CKCD=0.9)
output_ds.to_netcdf('/home/tkdals/homework_3/MPI_ex/data/800200_' + tc_name + '_output_durSST_donut_ori.nc')

# %%
output_ds = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/data/800200_' + tc_name + '_output_durSST_donut_ori.nc')
#%%
output_ds
#%%
output_ds['vmax'] = output_ds['vmax'] * 1.94384 # m/s t Knots
output_ds
#%%
# =============================================
# Plot MPI, TC wind speed
# =============================================
dur_mpi = output_ds['vmax'].data

usa_wind = df_tc.usa_wind # unit : Knots
Tmix_list
Dmix_list
time_arr = np.arange(len(dt))
plt.figure(figsize=(10, 6))
plt.plot(time_arr, usa_wind, 'bo-', color='deeppink', label='usa_wind')
plt.plot(time_arr, dur_mpi, 'bo-', color='purple', label='MPI')
plt.ylabel('wind speed(Knots)')
plt.title(tc_name + 'MPI, usa_wind')
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)  # 회전 추가로 레이블이 겹치지 않게 함
plt.yticks(fontsize=9)
plt.legend(loc='upper left')
plt.show()

# %%
print(len(dt))
#%%
# =============================================
# Plot MPI, Mixing temperature
# =============================================
mixing_t_2 = input_ds.sst.data
mpi_2 = output_ds.vmax.data
time_arr = np.arange(0, len(dt))
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots()
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)
ax1.set_ylim(0, 150)
ax1.set_ylabel('mpi(knots)')
ax1.plot(time_arr, mpi_2, 'bo-', color='red', label='mpi')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.set_ylabel('mixing temperature(degreeC)')
ax2.set_ylim(0, 30)
ax2.plot(time_arr, mixing_t_2, 'bo-', color='blue', label='mixing temp')
ax2.legend(loc='upper left')

# plt.title('Old mixing temperature, MPI 6 hours interval')
plt.title('Choiwan Mixing temperature, MPI')
plt.show()
# %%
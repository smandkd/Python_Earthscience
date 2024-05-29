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
path_data = '/home/tkdals/homework_3/MPI_ex/data/' # change path for your environment
path_fig = '/home/tkdals/homework_3/MPI_ex/figure/' # change path for your environment

tc_name = 'sanba' # You can only write tc between 2004 and 2020
tc_sid = b'2012254N09135' # sid 
tc_agency = b'jtwc_wp' # agency 
# sanba 2012254N09135
# haitang 2005192N22155
# saomai 2006216N07151
area_type = '500' # area_type is radius of input area. If you enter 500, then circle area with radius 500km will be made. 
# area_type = 'donut' for mixing temp, other input datas
# area_type = 'usa_rmw' for mixing temp
#%% 
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc') 
tc_dataset = mt.preprocess_IBTrACS(dataset, tc_sid, tc_agency)
tc_dataset
#%%
# ====================================================
# extracting tc date array, before 3 days date array 
# ====================================================
tc_date = tc_dataset['time']

pre_dt = mt.TC_pre_3_date(tc_dataset)
dt = mt.TC_present_date(tc_dataset)

formatted_dates = [f"{date.astype(object).month}/{date.astype(object).day}" for date in dt]
formatted_dates

tc_coords = mt.TC_usa_center_points(tc_dataset)
#%% 
# ==============================================================
# Preprocess Air Temp, Specific Hum, MSLP, SST from ERA5, OISST 
# dataset using pre_dt, dt array
# ==============================================================
years_months = sorted(set([(d.astype('datetime64[Y]').astype(int) + 1970, d.astype('datetime64[M]').astype(int) % 12 + 1) for d in dt]))
year = years_months[0][0]
month = years_months[0][1]

airt_path = f'/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_{year:04d}{month:02d}.nc'
shum_path = f'/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_{year:04d}{month:02d}.nc'
mslp_path = f'/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_{year:04d}{month:02d}.nc'
sst_path = f'/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.{year:04d}.nc'
airt_dataset = xr.open_dataset(airt_path)
shum_dataset = xr.open_dataset(shum_path)
mslp_dataset = xr.open_dataset(mslp_path)
sst_dataset = xr.open_dataset(sst_path)

level_arr = mt.sort_level_downgrade(airt_dataset) # for calculating mpi, reverse level array from ERA5 
preset_shum_dataset = shum_dataset.sel(time=dt, drop=True) 
days_3_before_oisst = sst_dataset.sel(time=pre_dt, drop=True)
days_3_before_mslp = mslp_dataset.sel(time=pre_dt, drop=True)
present_airt = airt_dataset.sel(time=dt, drop=True)

# ====================================================================
# Create Dataframe of TC. Values are time, usa_wind, usa_lat, usa_lon, 
# usa_r34, storm_speed, usa_rmw.
# ====================================================================
df_tc = mt.create_TC_dataframe(tc_dataset)
#%%
# ---------------------------------------------
# Specific Humidity
# kg kg**-1(kg/kg) to g/kg by multipling 1000
# ---------------------------------------------
shum_arr = []
shum_10_list = []
shum_arr, shum_10_list = mt.shum_mean(dt, tc_coords, preset_shum_dataset, area_type)

#%%
# -------------------------
# Sea Sureface Temperature
# -------------------------
sst_arr = []
haitang_rmw = tc_dataset.usa_rmw.data[0]
input_area_lat = []
input_area_lon = []

input_area_lon, input_area_lat, sst_arr = mt.sst_mean(pre_dt, tc_coords, days_3_before_oisst, area_type)
#%%
# =================================================
# Create a input area plot with Cartopy projection
# following tc track
# =================================================
flat_lon = np.concatenate(input_area_lon)
flat_lat = np.concatenate(input_area_lat)

max_lat = int(flat_lat.max() + 10)
max_lon = int(flat_lon.max() + 10)
min_lat = int(flat_lat.min() - 10)
min_lon = int(flat_lon.min() - 10)
print(min_lon, max_lon, min_lat, max_lat)

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgreen')
ax.add_feature(cfeature.OCEAN, facecolor='aqua')
ax.add_feature(cfeature.LAKES, facecolor='aqua')
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True)

# Plot points on the map
ax.scatter(flat_lon, flat_lat, s=10, color='red', marker='o', alpha=0.7, transform=ccrs.PlateCarree())

# Add title
plt.title('First Day input ear of ' + tc_name, weight='bold')
plt.savefig(path_fig + tc_name + '_' + area_type + '_day1_path.pdf')
plt.show()

#%%
# =================================================
# Create a input area plot with Cartopy projection
# first day of tc track
# =================================================

flat_lon = input_area_lon[0]
flat_lat = input_area_lat[0]

# Create a plot with Cartopy projection
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgreen')
ax.add_feature(cfeature.OCEAN, facecolor='aqua')
ax.add_feature(cfeature.LAKES, facecolor='aqua')
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True)

# Plot points on the map
ax.scatter(flat_lon, flat_lat, s=10, color='red', marker='o', alpha=0.7, transform=ccrs.PlateCarree())

# Add title
plt.title('Input Era following TC track of ' + tc_name, weight='bold')
plt.savefig(path_fig + tc_name + '_' + area_type + '_all_day_path.pdf')
plt.show()
#%%
# ---------------------------------------
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# ---------------------------------------
mslp_arr = []

mslp_arr = mt.mslp_mean(pre_dt, tc_coords, days_3_before_mslp, area_type)

#%%
# ----------------------------------------------------------
# Air Temperature 
# air temperature unit is K, so change to degC ( - 273.15 )
# ----------------------------------------------------------
airt_array = []
airt_10_list = []
airt_array, airt_10_list = mt.airt_mean(dt, tc_coords, present_airt, area_type)

# %%
# =====================================================
# Define dimensions, input variable and matching array 
# =====================================================
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

dataset = xr.Dataset(data_vars, coords=dims)
nc_path = path_data + tc_name + '_input_preSST_'+area_type+'.nc'
print(nc_path)
dataset.to_netcdf(path=nc_path)
input_ds = xr.open_dataset(nc_path)
input_ds

#%%
# ===============================
# run mpi calculation
# ===============================
output_ds = mpi.run_sample_dataset(input_ds, CKCD=0.9)
output_ds.to_netcdf(path_data + tc_name + '_output_preSST_'+area_type+'.nc')

#%%
input_ds = xr.open_dataset(path_data + tc_name + '_input_preSST_'+area_type+'.nc')
output_ds = xr.open_dataset(path_data + tc_name + '_output_preSST_'+area_type+'.nc')
input_ds
# output_ds
#%%
output_ds['vmax'] = output_ds['vmax'] * 1.94384 # m/s t Knots
output_ds
#%%
#%%
# =============================================
# Plot MPI, TC wind speed
# =============================================
dur_mpi = output_ds.vmax.data
usa_wind = df_tc.usa_wind # unit : Knots
time_arr = np.arange(len(dt))
plt.figure(figsize=(10, 6))
plt.plot(time_arr, usa_wind, 'bo-', color='deeppink', label='usa_wind')
plt.plot(time_arr, dur_mpi, 'bo-', color='purple', label='MPI')
plt.ylabel('wind speed(Knots)', weight='bold')
plt.title(tc_name + ' MPI, usa_wind', weight='bold')
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)  # 회전 추가로 레이블이 겹치지 않게 함
plt.yticks(fontsize=9)
plt.legend(loc='upper left')
plt.savefig(path_fig + tc_name + ' MPI_usawind_graph_'+area_type+'.pdf', dpi='figure')
plt.show()
# %%

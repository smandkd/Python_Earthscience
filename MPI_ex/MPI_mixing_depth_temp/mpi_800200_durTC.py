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
path_data = '/home/tkdals/homework_3/MPI_ex/data/'
path_fig = '/home/tkdals/homework_3/MPI_ex/figure/'
tc_name = 'haitang'
tc_sid = b'2005192N22155'
# sanba 2012254N09135
# haitang 2005192N22155
# saomai 2006216N07151
tc_agency = b'jtwc_wp'
area_type = 'donut'
mix_prof_type = 'donut'
# area_type = 'usa_rmw'

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
dt
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
df_tc
#%%
# ----------------------------
#            DAT
# ----------------------------

depth_list = [] # Dmix list

ecco2_lon = [] # longitude in TC radius
ecco2_lat = [] # latitude in TC radius
Tmix_list = []
Dmix_list = [] 
Theta_list = []
Salt_list = []
Dens_list = []
FT_list = []
Tx_list = []

Tmix_list, Dmix_list, ecco2_lon, ecco2_lat = mt.calculate_depth_mixing_d_t(df_tc, mix_prof_type)

print(f'Tmix : {np.round(Tmix_list, 2)}')
print(f'Dmix : {np.round(Dmix_list, 2)}')

#%%
flat_lon = np.concatenate(ecco2_lon)
flat_lat = np.concatenate(ecco2_lat)
# Create a plot with Cartopy projection
max_lat = flat_lat.max() + 10
max_lon = flat_lon.max() + 10
min_lat = flat_lat.min() - 10
min_lon = flat_lon.min() - 10

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
plt.title('First Day input area of ' + tc_name, weight='bold')
plt.savefig(path_fig + tc_name + '_' + area_type + '_day1_path.pdf')
plt.show()

#%%
flat_lon = ecco2_lon[0]
flat_lat = ecco2_lat[0]

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
# -------------------------------------------------
# Specific Humidity
# kg kg**-1(kg/kg) to g/kg by multipling 1000
# -------------------------------------------------

shum_arr = []
shum_10_list = []
shum_arr, shum_10_list = mt.shum_mean(dt, tc_coords, present_shum_dataset, area_type)

#%%
# --------------------------------------
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# --------------------------------------
mslp_arr = []
mslp_arr = mt.mslp_mean(pre_dt, tc_coords, days_3_before_mslp, area_type)

#%%
# ------------------------------------------------
# Air Temperature 
# t's unit is K, so change to degC ( - 273.15 )
# ------------------------------------------------
airt_array = []
airt_10_list = []
airt_array, airt_10_list = mt.airt_mean(dt, tc_coords, present_airt_dataset, area_type)
# %%

dims = dict(
    time = dt,
    level = level_arr
)

lsm_arr = np.ones((len(pre_dt), len(level_arr)))

data_vars = {
    'lsm': (['time', 'level'], lsm_arr),
    'airt': (['time', 'level'], airt_array),
    'shum': (['time', 'level'], shum_arr),
    'mslp': (['time'], mslp_arr),
    'sst' : (['time'], Tmix_list),
    'mixD' : (['time'], Dmix_list),
    'shum_10m': (['time'], shum_10_list),
    'airt_10m': (['time'], airt_10_list),
}

dataset = xr.Dataset(data_vars, coords=dims)
output_path = path_data + tc_name + '_output_durSST'+area_type+'_'+mix_prof_type+'.nc'
input_path = path_data + tc_name + '_input_durSST'+area_type+'_'+mix_prof_type+'.nc'
#%%
dataset.to_netcdf(path=input_path)
input_ds = xr.open_dataset(input_path)
input_ds

#%%

output_ds = mpi.run_sample_dataset(input_ds, CKCD=0.9)
output_ds.to_netcdf(output_path)

# %%
output_ds = xr.open_dataset(path_data + tc_name + '_output_durSST'+area_type+'_'+mix_prof_type+'.nc')
input_ds = xr.open_dataset(path_data + tc_name + '_input_durSST'+area_type+'_'+mix_prof_type+'.nc')
input_ds
#%%
output_ds['vmax'] = output_ds['vmax'] * 1.94384 # m/s t Knots
output_ds

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
plt.savefig('/home/tkdals/homework_3/MPI_ex/figure/' + tc_name + ' MPI_usawind_graph_'+area_type+'.pdf', dpi='figure')
plt.show()

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
ax1.set_ylabel('mpi(knots)', weight='bold')
ax1.plot(time_arr, mpi_2, 'bo-', color='red', label='mpi')
ax1.legend(loc='lower right')

ax2 = ax1.twinx()
ax2.set_ylabel('mixing temperature(degreeC)', weight='bold')
ax2.set_ylim(0, 30)
ax2.plot(time_arr, mixing_t_2, 'bo-', color='blue', label='mixing temp')
ax2.legend(loc='lower left')

plt.title(tc_name + ' Mixing temperature, MPI', weight='bold')
plt.show()
# %%
# =============================================
# Plot Mixing temperature, depth
# =============================================
input_ds = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/data/800200_' + tc_name + '_input_durSST_donut_'+area_type+'_prof.nc')
mixing_d = input_ds.mixD.data
mixing_t = input_ds.sst.data
time_arr = np.arange(0, len(dt))

plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots()
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)

ax1.set_ylim(0, 30)
ax1.set_ylabel('mixing temperature(degreeC)', weight='bold')
ax1.plot(time_arr, mixing_t, 'bo-', color='red', label='mixing t')
ax1.legend(loc='lower left')

ax2 = ax1.twinx()
ax2.set_ylim(0, 150)
ax2.set_ylabel('mixing depth(m)', weight='bold')
ax2.plot(time_arr, mixing_d, 'bo-', color='blue', label='mixing d')
ax2.legend(loc='lower right')

plt.title(tc_name + ' mixing temperature(degreeC),depth(m), 6 hours interval', weight='bold')
plt.savefig('/home/tkdals/homework_3/MPI_ex/figure/' + tc_name + ' mixing_t_d_graph_'+area_type+'.pdf', dpi='figure')
plt.show()
# %%

#%%
# =======================================================
#
#  태풍 반경, 중심 좌표 지도에 표시
#
# =======================================================
combined_ecco2_lat = np.concatenate(ecco2_lat)
combined_ecco2_lon = np.concatenate(ecco2_lon)
haitang_center_points_lon = df_tc.usa_lon
haitang_center_points_lat = df_tc.usa_lat
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set map extent to cover the area of interest
ax.set_extent([100, 180, 1, 40], crs=ccrs.PlateCarree())

# Add map features for clarity and context
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAND, facecolor='lightgreen', edgecolor='black', zorder=1)

# Define the coordinates to be plotted (these are sample coordinates)
# Plot the points on the map with a different color and size for distinction
ax.scatter(combined_ecco2_lon, combined_ecco2_lat, color='red', s=50, transform=ccrs.PlateCarree(), label='input area')
ax.scatter(haitang_center_points_lon, haitang_center_points_lat, color='black', s=5, transform=ccrs.PlateCarree(), label='Typhoon Track Centers')

# Add titles, labels, and a legend
ax.set_title("Typhoon Haitang Affected Coordinates")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.legend(loc='upper left')

# Save and show the plot
plt.savefig("haitang_track_intensity_modified.pdf")
plt.show()
#%%
"""# %%
# ===========================================================
#
#  Salinity, Potential temperature interpolation,
#  5m 데이터 확인
#
# ===========================================================
new_longitude = np.arange(-179.875, 180.125, 0.25)
ecco2_salt_ds = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/SALT/2005/SALT.1440x720x50.20050707.nc')
ecco2_theta_ds = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/THETA/2005/THETA.1440x720x50.20050707.nc')
mt.replace_longitude(ecco2_salt_ds, new_longitude)
mt.replace_longitude(ecco2_theta_ds, new_longitude)
lat = ecco2_lon[0] # first latitude of haitang
lon = ecco2_lat[0] # first longitude of haitang
ecco2_theta_0 = ecco2_theta_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
ecco2_salt_0 = ecco2_salt_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
theta = ecco2_theta_0.THETA.data[0] 
salt = ecco2_salt_0.SALT.data[0] 
depth = ecco2_theta_0.DEPTH_T.data
theta_5m_arr = theta[0]

the_list = []
theta_list = []
Theta_list = []

for i in range(len(ecco2_lat)):
    lat = ecco2_lat[i] # first latitude of haitang
    lon = ecco2_lon[i] # first longitude of haitang
    ecco2_theta_0 = ecco2_theta_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
    ecco2_salt_0 = ecco2_salt_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
    theta = ecco2_theta_0.THETA.data[0] 
    salt = ecco2_salt_0.SALT.data[0] 

    theta_5m_arr = theta[0]
    
    the_list = []
    theta_list = []
    Theta_list = []
    count = 0 
    
    for row in theta_5m_arr:
        print(row)
        theta_list.append(np.mean(row))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    
    print('====================================================')
new_depth = np.arange(min(depth), max(depth), 1)  # 1m 간격
# 1차원 보간 객체 생성
interp_theta = interpolate.interp1d(depth, theta, kind='linear', fill_value='extrapolate')
interp_salt = interpolate.interp1d(depth, salt, kind='linear', fill_value='extrapolate')
# 새로운 깊이에 대한 온도 예측
new_theta = interp_theta(new_depth)
new_salt = interp_salt(new_depth)
new_theta[:3] = new_theta[4]
new_salt[:3] = new_salt[4]

# =====================================================
#  Interpolated salinity, potential temperature graph
# =====================================================
plt.plot(depth, theta, 'o', label='Original Data')  # 원래 데이터
plt.plot(new_depth, new_theta, '-', label='Interpolated Data')  # 보간된 데이터
plt.xlabel('Depth (m)')
plt.ylabel('Potential Temp (degree celcius)')
plt.title('Potential Temp vs. Depth (Interpolated)')
plt.legend()
plt.show()

plt.plot(depth, salt, 'o', label='Original Data')  # 원래 데이터
plt.plot(new_depth, new_salt, '-', label='Interpolated Data')  # 보간된 데이터
plt.xlabel('Depth (m)')
plt.ylabel('Salinity (PSU)')
plt.title('Salinity vs. Depth (Interpolated)')
plt.legend()
plt.show()
airt_1 = airt[0]
airt_last = airt[-1]

level = [1000,  925,  850,  700,  600,  500,  400,  300,  250,  200,  100,   70,
         50,   30,   20,   10]

plt.plot( airt_1, level[::-1], 'bo-', color='blue')
plt.ylabel('level')
plt.xlabel('airt(degreeC)')
plt.title('airt')
plt.yticks(fontsize=8)
plt.show()
#%%
level = [1000,  925,  850,  700,  600,  500,  400,  300,  250,  200,  100,   70,
         50,   30,   20,   10]
plt.plot(airt_last, level[::-1], 'bo-', color='blue')
plt.ylabel('level')
plt.xlabel('airt(degreeC)')
plt.title('airt')
plt.yticks(fontsize=8)
plt.show()
# %%

time_arr = np.arange(0, 20)
plt.plot(time_arr, mslp, 'bo-', color='red')
plt.ylabel('pressure(hPa)')
plt.title('mean sea level pressure')
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=8)  # 회전 추가로 레이블이 겹치지 않게 함
plt.yticks(fontsize=8)
plt.show()

# %%
level = [1000,  925,  850,  700,  600,  500,  400,  300,  250,  200,  100,   70,
         50,   30,   20,   10]

plt.plot(shum[0], level[::-1], 'bo-', color='deeppink')
plt.ylabel('level')
plt.xlabel('humidity(g/kg)')
plt.title('specific humidity')
plt.yticks(fontsize=8)
plt.show()
#%%
level = [1000,  925,  850,  700,  600,  500,  400,  300,  250,  200,  100,   70,
         50,   30,   20,   10]
plt.plot( shum[-1], level[::-1], 'bo-', color='deeppink')
plt.ylabel('level')
plt.xlabel('humidity(g/kg)')
plt.title('specific humidity')
plt.yticks(fontsize=8)
plt.show()
"""
# %%

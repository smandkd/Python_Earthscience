#%%
import mixingdt_mpi.methods as mt 
import mixingdt_mpi.mpi as mpi 

import scipy.interpolate as interpolate

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import shapely.geometry as sgeom

import warnings
warnings.filterwarnings(action='ignore')
# %%
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
saomai_dataset = mt.preprocess_IBTrACS(dataset, b'2006216N07151', b'jtwc_wp')

saomai_date = saomai_dataset['time']

pre_dt = mt.TC_pre_3_date(saomai_dataset)
dt = mt.TC_present_date(saomai_dataset)

SAOMAI_coords = mt.TC_usa_center_points(saomai_dataset)

# Haitang usa latitude, longitude 
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200608.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200608.nc')
mslp_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200608.nc')
sst_dataset = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2006.nc')

level_arr = mt.sort_level_downgrade(airt_dataset)
#%%
present_shum_dataset = shum_dataset.sel(time=dt, drop=True)
days_3_before_oisst = sst_dataset.sel(time=pre_dt, drop=True)
days_3_before_mslp = mslp_dataset.sel(time=pre_dt, drop=True)
present_airt_dataset = airt_dataset.sel(time=dt, drop=True)
#%%
saomai_dataset
df_saomai = mt.create_TC_dataframe(saomai_dataset)
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

Tmix_list, Dmix_list, Theta_list, Salt_list, Dens_list, FT_list, Tx_list = mt.calculate_depth_mixing_d_t(df_saomai)

print(f'Tmix : {np.round(Tmix_list, 2)}')
print(f'Dmix : {np.round(Dmix_list, 2)}')
print(f'Potential temperature : {Theta_list}')
print(f'Salinity : {Salt_list}')
print(f'Density : {Dens_list}')
print(f'Residence time : {FT_list}')
print(f'Wind stress : {Tx_list}')
#%%
# -------------------
# Specific Humidity
# -------------------

shum_mean_arr = []
shum_10_list = []
shum_mean_arr, shum_10_list = mt.shum_donut_mean(SAOMAI_coords, present_shum_dataset)


#%%
# -----------------------
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# -----------------------
mslp_daily_averages = []

mslp_daily_averages = mt.mslp_donut_mean(pre_dt, SAOMAI_coords, days_3_before_mslp)

#%%
# -----------------------
# Air Temperature 
# t's unit is K, so change to degC ( - 273.15 )
# -----------------------
airt_array = []
airt_10_list = []
airt_array, airt_10_list = mt.airt_donut_mean(dt, SAOMAI_coords, present_airt_dataset)
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
    'salt' : (['time'], Salt_list),
    'theta' : (['time'], Theta_list),
    'shum_10m': (['time'], shum_10_list),
    'airt_10m': (['time'], airt_10_list),
    'density': (['time'], Dens_list),
    'resitime': (['time'], FT_list),
    'windstress':(['time'], Tx_list)
}

dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_saomai_input_durSST_donut.nc'
dataset.to_netcdf(path=nc_path)
#%%
nc_path = '/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_saomai_input_durSST_donut.nc'
input_ds = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_saomai_input_durSST_donut.nc')
input_ds
#%%
output_ds = mpi.run_sample_dataset(nc_path, CKCD=0.9)
output_ds.to_netcdf('800200_saomai_output_durSST_donut.nc')

# %%
output_ds = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/800200_saomai_output_durSST_donut.nc')
#%%
output_ds
#%%
output_ds['vmax'] = output_ds['vmax'] * 1.94384 # m/s t Knots
output_ds
#%%
dur_mpi = output_ds.vmax.data
dat = output_ds.sst.data
time = output_ds.time.data
msl = output_ds.mslp.data
airt = output_ds.airt.data
hum = output_ds.shum.data
#%%
ocean_potent_temp = input_ds.theta.data
salinity_list = input_ds.salt.data
shum_10_list = input_ds.shum_10m.data
airt_10_list = input_ds.airt_10m.data
dens_list = input_ds.density.data
residence_time_list = input_ds.resitime.data
depth_list = input_ds.mixD.data
wind_stress_list = input_ds.windstress.data
#%%
# =============================================
# DAT, MPI 
# =============================================
time_arr = np.arange(0, 18)
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('MPI(Knots)')
ax1.plot(time_arr, dur_mpi, 'bo-', color='green', label='mpi')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.set_ylabel('DAT(degreeC)')
ax2.plot(time_arr, dat, 'bo-', color='deeppink', label='sst')
ax2.legend(loc='lower right')

plt.xticks(np.arange(1, 18, step=1))
plt.show()

# %%
# ===============================
#    한개만 그리기 variable D
# ===============================
time_arr = np.arange(0, 18)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, depth_list, 'bo-', color='green')
plt.xticks(np.arange(0, 18, step=1))
plt.xlabel('time')
plt.ylabel('saomai mixing depth(m)')
plt.title('Saomai mixing depth(m)')
plt.show()

#%%
# ===============================
#        한개만 그리기 DAT
# ===============================
time_arr = np.arange(0, 18)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, Tmix_list, 'bo-')
plt.xticks(np.arange(0, 18, step=1))
plt.xlabel('time')
plt.ylabel('saomai Depth-averaged temperature(degreeC)')
plt.title('Saomai Depth-averaged temperature(degreeC)')
plt.show()


#%%
# ====================================================
#        Potential Temperature, Salinity
#  density, storm speed, residence time, wind stress
# ====================================================
time_arr = np.arange(0, 18)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, salinity_list, 'bo-', color='deeppink')
plt.xticks(np.arange(0, 18, step=1))
plt.xlabel('time')
plt.ylabel('Potential temperature(*C)')
plt.title('Saomai ocean potential temperature(depth 5m)')
plt.show()
#%%
plt.figure(figsize=(10, 6))
plt.plot(time_arr, Salt_list, 'bo-', color='green')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('Salinity(PSU)')
plt.title('Saomai Salinity (depth 5m)')
plt.show()
#%%
time_arr = np.arange(0, 18)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, dens_list, 'bo-', color='black')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('Density(kg/m^3)')
plt.title('Saomai density (depth 5m)')
plt.show()
#%%
time_arr = np.arange(1, 19)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, df_saomai.storm_speed, 'bo-', color='deeppink')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('Storm speed(Knots)')
plt.title('Saomai storm speed')
plt.show()
#%%
time_arr = np.arange(1, 19)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, residence_time_list, 'bo-', color='red')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('residence time(sec)')
plt.title('Saomai residence time')
plt.show()
#%%
time_arr = np.arange(1, 19)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, wind_stress_list, 'bo-', color='purple')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('wind stress(N/m^2)')
plt.title('Saomai wind stress')
plt.show()

# %%
# ===============================
#        4개 그리기
# ===============================

time_arr = np.arange(0, 18)
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(time_arr, dur_mpi, 'bo-', color='green')
plt.title('DUR_MPI(Knots)')
plt.ylabel('mpi')

plt.subplot(2, 2, 2)
plt.plot(time_arr, dat, 'bo-', color='red')
plt.title('Depth-averaged temperature(*C)')
plt.ylabel('dat')

plt.subplot(2, 2, 3)
plt.plot(time_arr, airt_10_list, 'bo-', color='deeppink')
plt.title('10m level Air Temperature(*C)')
plt.ylabel('hum')

plt.subplot(2, 2, 4)
plt.plot(time_arr, shum_10_list, 'bo-', color='blue')
plt.title('10m level Specific Humidity(g/kg)')
plt.ylabel('airt')


# %%
# ===============================
#   dur, pre 두 그래프 그리기
# ===============================
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)

fig, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('mpi(m/s)')
ax1.plot(time_arr, dur_mpi, 'bo-', color='green', label='dur_mpi')
ax1.legend(loc='upper right')

# %%
new_longitude = np.arange(-179.875, 180.125, 0.25)
ecco2_salt_ds = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/SALT/2006/SALT.1440x720x50.20060804.nc')
ecco2_theta_ds = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/THETA/2006/THETA.1440x720x50.20060804.nc')
mt.replace_longitude(ecco2_salt_ds, new_longitude)
mt.replace_longitude(ecco2_theta_ds, new_longitude)

#%%
# ==========================================================
#
# Potential temperature, Salinity 수심 5m 깊이 데이터
#
# ==========================================================

lat = df_saomai.usa_lat[0] # first latitude of haitang
lon = df_saomai.usa_lon[0] # first longitude of haitang
ecco2_theta_0 = ecco2_theta_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
ecco2_salt_0 = ecco2_salt_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
theta = ecco2_theta_0.THETA.data[0] 
salt = ecco2_salt_0.SALT.data[0] 
depth = ecco2_theta_0.DEPTH_T.data
theta_5m_arr = theta[0]

the_list = []
theta_list = []
Theta_list = []

for i in range(len(ecco2_saomai_lat)):
    lat = ecco2_saomai_lat[i] # first latitude of haitang
    lon = ecco2_saomai_lon[i] # first longitude of haitang
    ecco2_theta_0 = ecco2_theta_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
    ecco2_salt_0 = ecco2_salt_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
    theta = ecco2_theta_0.THETA.data[0] 
    salt = ecco2_salt_0.SALT.data[0] 

    theta_5m_arr = theta[0]
    print(theta_5m_arr)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    the_list = []
    theta_list = []
    Theta_list = []
    count = 0 
    
    for row in theta_5m_arr:
        for x in row: 
            the_list.append(x)
        theta_list.append(np.nanmean(the_list))
    
    Theta_list.append(theta_list)

Theta_list

# %%
# ============================================================
#
# Interpolation 그래프 그리기
#
# ============================================================
new_depth = np.arange(min(depth), max(depth), 1)  # 1m 간격

# 1차원 보간 객체 생성
interp_theta = interpolate.interp1d(depth, theta, kind='linear', fill_value='extrapolate')
interp_salt = interpolate.interp1d(depth, salt, kind='linear', fill_value='extrapolate')

# 새로운 깊이에 대한 온도 예측
new_theta = interp_theta(new_depth)
new_salt = interp_salt(new_depth)

new_salt[:3] = new_salt[4]
new_theta[:3] = new_theta[4]

# %%
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

#%%
# ============================================
# 
# Saomai 태풍 트랙 표시 
# 
# ============================================
intensities = saomai_dataset.usa_wind.data[0]
lat = np.array(np.round(saomai_dataset.usa_lat[0],2))
lon = np.array(np.round(saomai_dataset.usa_lon[0],2))
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())
ax.set_extent([100, 180, 1, 40], crs=crs.PlateCarree())
states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
ax.add_feature(states_provinces, edgecolor="gray")
ax.add_feature(cfeature.LAND, zorder=1, edgecolor="k")
ax.add_feature(cfeature.OCEAN)
ax.gridlines(draw_labels=True)

track = sgeom.LineString(zip(lon, lat))
track_buffer = track.buffer(2)

def colorize_state(geometry):
    facecolor = (0.9375, 0.9375, 0.859375)
    if geometry.intersects(track):
        facecolor = 'red'
    elif geometry.intersects(track_buffer):
        facecolor = '#FF7E00'
    return {'facecolor': facecolor, 'edgecolor': 'black'}


ax.add_geometries([track], crs.PlateCarree(),
                    facecolor='none', edgecolor='k')

ax.plot(lon[0], lat[0], 'o', transform=crs.PlateCarree(), color='red', markersize=5)  # Start point
ax.text(lon[0], lat[0], str(intensities[0]), transform=crs.PlateCarree(), fontsize=9, ha='right', va='center')

ax.plot(lon[-1], lat[-1], 'o', transform=crs.PlateCarree(), color='blue', markersize=5)  # End point
ax.text(lon[-1], lat[-1], str(intensities[-1]), transform=crs.PlateCarree(), fontsize=9, ha='right', va='center')

plt.savefig("saomai_track_intensity.pdf")
plt.show()

# ======================================================
#
# 태풍 반경, 중심좌표 그리기 
#
# ======================================================

combined_ecco2_lat = np.concatenate(ecco2_saomai_lat)
combined_ecco2_lon = np.concatenate(ecco2_saomai_lon)
saomai_center_points_lon = df_saomai.usa_lon
saomai_center_points_lat = df_saomai.usa_lat

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())

# Set map extent to cover the area of interest
ax.set_extent([100, 180, 1, 40], crs=crs.PlateCarree())

# Add map features for clarity and context
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAND, facecolor='lightgreen', edgecolor='black', zorder=1)

# Define the coordinates to be plotted (these are sample coordinates)
# Plot the points on the map with a different color and size for distinction
ax.scatter(combined_ecco2_lat, combined_ecco2_lat, color='red', s=50, transform=crs.PlateCarree(), label='Typhoon Radius Points')
ax.scatter(saomai_center_points_lon, saomai_center_points_lat, color='black', s=5, transform=crs.PlateCarree(), label='Typhoon Track Centers')

# Add titles, labels, and a legend
ax.set_title("Typhoon Saomai Affected Coordinates")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.legend(loc='upper left')

# Save and show the plot
plt.savefig("saomai_track_intensity_modified.pdf")
plt.show()
# %%

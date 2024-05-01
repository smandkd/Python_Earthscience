#%%
import MPI_ex.MPI_mixing_depth_temp.mixing_dt_mpi.methods as mt 
import MPI_ex.MPI_mixing_depth_temp.mixing_dt_mpi.mpi as mpi 

import scipy.interpolate as interpolate

import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature

import warnings
warnings.filterwarnings(action='ignore')

# %%
haitang_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/data/haitang_2005.nc')

haitang_date = haitang_dataset['time']
pre_dt = mt.TC_pre_3_date(haitang_date)
dt = mt.TC_present_date(haitang_date)
#%%
formatted_dates = [f"{date.astype(object).month}/{date.astype(object).day}" for date in dt]
formatted_dates
# %%
HAITANG_coords = mt.TC_usa_center_points(haitang_dataset)

# %%
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')
# %%
level_arr = mt.sort_level_downgrade(airt_dataset)

# %%
preset_shum_dataset = shum_dataset.sel(time=dt, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=pre_dt, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=pre_dt, drop=True)
present_airt = airt_dataset.sel(time=dt, drop=True)

# %%
df_haitang = mt.create_TC_dataframe(haitang_dataset)
#%%
# ----------------------------
#            DAT
# ----------------------------
depth_list = [] # Dmix list


ecco2_haitang_lon = [] # longitude in TC radius
ecco2_haitang_lat = [] # latitude in TC radius
Tmix_list = []
Dmix_list = [] 
Theta_list = []
Salt_list = []
Dens_list = []
FT_list = []
Tx_list = []

Tmix_list, Dmix_list, Theta_list, Salt_list, Dens_list, FT_list, Tx_list = mt.calculate_depth_mixing_d_t(df_haitang)

print(f'Tmix : {np.round(Tmix_list, 2)}')
print(f'Dmix : {np.round(Dmix_list, 2)}')
print(f'Potential temperature : {Theta_list}')
print(f'Salinity : {Salt_list}')
print(f'Density : {Dens_list}')
print(f'Residence time : {FT_list}')
print(f'Wind stress : {Tx_list}')

#%%
# ----------------------------------------------
#
# Specific Humidity
# kg kg**-1(kg/kg) to g/kg by multipling 1000
#
# ----------------------------------------------
shum_mean_arr = []
shum_10_list = []
shum_mean_arr, shum_10_list = mt.shum_donut_mean(HAITANG_coords, preset_shum_dataset)
#%%
# -----------------------
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# -----------------------
mslp_daily_averages = []

mslp_daily_averages = mt.mslp_donut_mean(pre_dt, HAITANG_coords, days_3_before_mslp)

#%%
# -----------------------
# Air Temperature 
# t's unit is K, so change to degC ( - 273.15 )
# -----------------------
airt_array = []
airt_10_list = []
airt_array, airt_10_list = mt.airt_donut_mean(dt, HAITANG_coords, present_airt)

# %%
dims = dict(
    time = dt,
    level = level_arr
)

# %%
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
#%%
input_path = '/home/tkdals/homework_3/MPI_ex/data/800200_haitang_input_durSST_donut.nc'
input_path_2 = '/home/tkdals/homework_3/MPI_ex/data/800200_haitang_input_durSST_donut_2.nc'
output_path = '/home/tkdals/homework_3/MPI_ex/data/800200_haitang_output_durSST_donut.nc'
output_path_2 = '/home/tkdals/homework_3/MPI_ex/data/800200_haitang_output_durSST_donut_2.nc'
#%%

dataset = xr.Dataset(data_vars, coords=dims)
dataset.to_netcdf(path=input_path)
#%%
ds = mpi.run_sample_dataset(input_path_2, CKCD=0.9)
ds.to_netcdf(output_path_2)

# %%
input_ds_2 = xr.open_dataset(input_path_2)
input_ds = xr.open_dataset(input_path)

output_ds = xr.open_dataset(output_path)
output_ds_2 = xr.open_dataset(output_path_2)
#%%
output_ds_2['vmax'] = output_ds_2['vmax'] * 1.94384 # m/s t0 Knots
#%%
output_ds['vmax'] = output_ds['vmax'] * 1.94384 # m/s t0 Knots
#%%
output_ds

#%%
dur_mpi = output_ds.vmax.data
dat = output_ds.sst.data
time = output_ds.time.data
mslp = output_ds.mslp.data
airt = output_ds.airt.data
hum = output_ds.shum.data
#%%
ocean_potent_temp = input_ds.theta.data
shum_10_list = input_ds.shum_10m.data
airt_10_list = input_ds.airt_10m.data
residence_time_list = input_ds.resitime.data
wind_stress_list = input_ds.windstress.data
mixing_depth_list = input_ds.mixD.data
theta_list = input_ds.theta.data
#%%
mixing_t_2 = input_ds_2.sst.data
mixing_t = input_ds.sst.data
mixing_d = input_ds.mixD.data
mixing_d_2 = input_ds_2.mixD.data
windstr = input_ds.windstress.data
windstr_2 = input_ds_2.windstress.data
mpi_2 = output_ds_2.vmax.data
mpi_1 = output_ds.vmax.data
#%%
mpi_2
#%%
len(mpi_1)
#%%
time_arr = np.arange(0, 20)

plt.plot(time_arr, mixing_d, 'bo-', color='green', label='old mixing depth')
plt.plot(time_arr, mixing_d_2, 'bo-',color='deeppink', label='new mixing depth')
plt.ylabel('depth(m)')
plt.title('Haitang mixing depth(m), 6 hours interval')
plt.legend(loc='upper left')
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)  # 회전 추가로 레이블이 겹치지 않게 함
plt.yticks(fontsize=9)
plt.xticks(np.arange(0, 20, step=1))
plt.show()

#%%
time_arr = np.arange(0, 20)
plt.plot(time_arr, mixing_t, 'bo-', color='green', label='old mixing temperature')
plt.plot(time_arr, mixing_t_2,'bo-', color='deeppink', label='new mixing temperature')
plt.ylabel('temperature(degreeC)')
plt.title('Haitang mixing temperature(degreeC), 6 hours interval')
plt.legend(loc='upper left')
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)  # 회전 추가로 레이블이 겹치지 않게 함
plt.yticks(fontsize=9)
plt.xticks(np.arange(0, 20, step=1))
plt.show()
#%%
time_arr = np.arange(0, 20)
plt.plot(time_arr, windstr, color='green', label='old wind stress')
plt.plot(time_arr, windstr_2, color='deeppink', label='new wind stress')
plt.ylabel('wind stress(N/m^2)')
plt.title('Haitang wind stress(N/m^2), 6 hours interval')
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=8)  # 회전 추가로 레이블이 겹치지 않게 함
plt.yticks(fontsize=8)
plt.legend(loc='upper left')
plt.show()
#%%
time_arr = np.arange(0, 20)
plt.plot(time_arr, mpi_1, 'bo-', color='green', label='old MPI')
plt.plot(time_arr, mpi_2, 'bo-', color='deeppink', label='new MPI')
plt.ylabel('MPI(Knots)')
plt.title('Haitang MPI(Knots), 6 hours interval')
plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=9)  # 회전 추가로 레이블이 겹치지 않게 함
plt.yticks(fontsize=9)
plt.legend(loc='upper left')
plt.show()

# %%
# ===============================
#    한개만 그리기 variable D
# ===============================
time_arr = np.arange(0, 20)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, mixing_depth_list, 'bo-', color='blue')
plt.xticks(np.arange(0, 20, step=1))
plt.ylim(0, 450)
plt.xlabel('time')
plt.ylabel('haitang mixing depth(m)')
plt.title('Haitang mixing depth(m)')
plt.show()

#%%
# ===============================
#    Wind stress
# ===============================
time_arr = np.arange(0, 20)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, wind_stress_list, color='green')
plt.xlabel('time')
plt.ylabel('wind stress')
plt.title('wind stress')
plt.show()

#%%
# ===============================
#        한개만 그리기 DAT
# ===============================
time_arr = np.arange(0, 20)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, dat, 'bo-', color="red")
plt.xticks(np.arange(0, 20, step=1))
plt.ylim(20, 30)
plt.xlabel('time')
plt.ylabel('haitang Depth-averaged temperature(degreeC)')
plt.title('Haitang Depth-averaged temperature(degreeC)')
plt.show()

#%%

# ========================================
#          DAT, MPI 그래프 
# ========================================

time_arr = np.arange(0, 20)
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('mpi(knots)')
ax1.plot(time_arr, mpi_2, 'bo-', color='red', label='mpi')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.set_ylabel('dat(degree celcius)')
ax2.plot(time_arr, mixing_t_2, 'bo-', color='blue', label='mixing temp')
ax2.legend(loc='upper left')

plt.xticks(ticks=np.arange(len(formatted_dates)), labels=formatted_dates, rotation=45, fontsize=8) 
plt.title('New mixing temperature, MPI 6 hours interval')
plt.show()
# %%
# ===================================================
#        Potential Temperature, Salinity
#  density, storm speed, residence time, wind stress
# ====================================================
#%%
plt.figure(figsize=(10, 6))
plt.plot(time_arr, Theta_list, 'bo-', color='deeppink')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('Potential temperature(*C)')
plt.title('Haitang ocean potential temperature(depth 5m)')
plt.show()
#%%
plt.figure(figsize=(10, 6))
plt.plot(time_arr, Salt_list, 'bo-', color='green')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('Salinity(PSU)')
plt.title('Haitang Salinity (depth 5m)')
plt.show()
#%%
time_arr = np.arange(1, 21)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, Dens_list, 'bo-', color='deeppink')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('density(kg/m^3)')
plt.title('Haitang density (depth 5m)')
plt.show()
#%%

time_arr = np.arange(1, 21)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, theta_list, 'bo-', color='deeppink')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('storm speed(Knots)')
plt.title('Haitang storm speed')
plt.show()

#%%
time_arr = np.arange(1, 21)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, residence_time_list, 'bo-', color='deeppink')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('residence time(sec)')
plt.title('Haitang residence time')
plt.show()
#%%
time_arr = np.arange(1, 21)
plt.figure(figsize=(10, 6))
plt.plot(time_arr, wind_stress_list, 'bo-', color='purple')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('time')
plt.ylabel('wind stress(N/m^2)')
plt.title('Haitang wind stress')
plt.show()
# %%
# ===============================
#        4개 그리기
# ===============================
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

ax1.plot(time_arr, pre_mpi, 'bo-', color='deeppink', label='pre_mpi')
ax1.legend(loc='lower right')

plt.xticks(np.arange(1, 21, step=1))
plt.show()

#   
#
# ===========================================================
#
#  Salinity, Potential temperature interpolation,
#  5m 데이터 확인
#
# ===========================================================
#
#
# %%
new_longitude = np.arange(-179.875, 180.125, 0.25)
ecco2_salt_ds = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/SALT/2005/SALT.1440x720x50.20050707.nc')
ecco2_theta_ds = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/THETA/2005/THETA.1440x720x50.20050707.nc')
mt.replace_longitude(ecco2_salt_ds, new_longitude)
mt.replace_longitude(ecco2_theta_ds, new_longitude)
lat = ecco2_haitang_lat[0] # first latitude of haitang
lon = ecco2_haitang_lon[0] # first longitude of haitang
ecco2_theta_0 = ecco2_theta_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
ecco2_salt_0 = ecco2_salt_ds.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True)
theta = ecco2_theta_0.THETA.data[0] 
salt = ecco2_salt_0.SALT.data[0] 
depth = ecco2_theta_0.DEPTH_T.data
theta_5m_arr = theta[0]

the_list = []
theta_list = []
Theta_list = []

for i in range(len(ecco2_haitang_lat)):
    lat = ecco2_haitang_lat[i] # first latitude of haitang
    lon = ecco2_haitang_lon[i] # first longitude of haitang
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
#%%
# =======================================================
#
#  태풍 반경, 중심 좌표 지도에 표시
#
# =======================================================
combined_ecco2_lat = np.concatenate(ecco2_haitang_lat)
combined_ecco2_lon = np.concatenate(ecco2_haitang_lon)
# %%
haitang_center_points_lon = df_haitang.usa_lon
haitang_center_points_lat = df_haitang.usa_lat
# %%
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
ax.scatter(combined_ecco2_lon, combined_ecco2_lat, color='red', s=50, transform=crs.PlateCarree(), label='Typhoon Radius Points')
ax.scatter(haitang_center_points_lon, haitang_center_points_lat, color='black', s=5, transform=crs.PlateCarree(), label='Typhoon Track Centers')

# Add titles, labels, and a legend
ax.set_title("Typhoon Haitang Affected Coordinates")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.legend(loc='upper left')

# Save and show the plot
plt.savefig("haitang_track_intensity_modified.pdf")
plt.show()

#%%
import MPI_ex.python_sang_ex.sang_ex.mixingdt_mpi.methods as mt 

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
#%% Extracting Haitang from IBTrACS
haitang_dataset = xr.open_dataset('/home/tkdals/homework_3/MPI_ex/python_sang_ex/sang_ex/haitang_2005.nc')
#%%
haitang_dataset
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
pre_dt
#%%
lat = np.array(np.round(haitang_dataset.usa_lat[0],2))
lon = np.array(np.round(haitang_dataset.usa_lon[0],2))
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lon, lat))


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
oisst_DATA.sel(time='2005-07-07T00:00:00.000000000', lon = 151.89, lat = 23.0, method='nearest', drop=True)

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
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.stock_img()
ax.coastlines()
ax.scatter(donut_lon[0], donut_lat[0], color='blue', s=0.5, transform=crs.PlateCarree())
plt.savefig("haitang_800200.pdf")
plt.show()

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
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
print(sst_daily_averages)

#%%
sst_daily_averages
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

plt.figure(figsize=(10, 7))
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
#%%
# ============================================
# Haitang 태풍 트랙 표시 
# ============================================
intensities = haitang_dataset.usa_wind.data[0]
lat = np.array(np.round(haitang_dataset.usa_lat[0],2))
lon = np.array(np.round(haitang_dataset.usa_lon[0],2))
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

plt.savefig("haitang_track_intensity.pdf")
plt.show()


"""
inner_coords = set(zip(ex_lon_arr[0], ex_lat_arr[0]))

def is_close(point, other_points_set, threshold=1.3):
    return any(np.linalg.norm(np.array(point) - np.array(other_point)) < threshold 
               for other_point in other_points_set)
# We can now use this function to filter the outer points
# Assuming 'threshold' is the distance within which points are considered 'close'
threshold_distance = 1.3  # Define your own threshold here

# Now, filter out the close points
filtered_outer_coords = [(lon, lat) for lon, lat in zip(outer_lon_arr[0], outer_lat_arr[0])
                         if not is_close((lon, lat), inner_coords, threshold_distance)]
filtered_outer_coords_array = np.array(filtered_outer_coords)
# Split the filtered coordinates back into separate arrays
filtered_outer_lon_arr, filtered_outer_lat_arr = filtered_outer_coords_array[:, 0], filtered_outer_coords_array[:, 1]

print(len(filtered_outer_lat_arr), len(filtered_outer_lon_arr))

filtered_outer_lon_arr, filtered_outer_lat_arr
print(len(filtered_outer_lat_arr), len(filtered_outer_lon_arr))
#%%

inner_coords = set(zip(ex_lon_arr[0], ex_lat_arr[0]))

# 외부 원 좌표와 내부 원 좌표 사이의 거리를 계산하는 함수입니다.
def calculate_distance(outer_coord, inner_coords):
    distances = [np.sqrt((outer_coord[0] - ic[0])**2 + (outer_coord[1] - ic[1])**2) for ic in inner_coords]
    return np.min(distances)

# 중앙 영역에 있는 점들을 배제하고 외부 원 좌표만 남기는 코드입니다.
# 임계값은 내부 원과의 최소 거리를 정의합니다.
threshold_distance = 0.1  # '가깝다'고 정의할 임계 거리입니다.

filtered_outer_coords = np.array([
    coord for coord in zip(outer_lon_arr[0], outer_lat_arr[0])
    if calculate_distance(coord, inner_coords) > threshold_distance
])

# 필터링된 외부 좌표를 경도와 위도 배열로 분리합니다.
filtered_outer_lon_arr, filtered_outer_lat_arr = filtered_outer_coords[:, 0], filtered_outer_coords[:, 1]

# 필터링된 결과를 출력합니다.
print(len(filtered_outer_lat_arr), len(filtered_outer_lon_arr))


center_lat = np.array(np.round(haitang_dataset.usa_lat[0],2))[0]
center_lon = np.array(np.round(haitang_dataset.usa_lon[0],2))[0]

outer_lon_arr = outer_lon_arr[0]
outer_lat_arr = outer_lat_arr[0]

print(f'outer lon : {outer_lon_arr}, {len(outer_lon_arr)}')
print(f'outer lat : {outer_lat_arr}, {len(outer_lat_arr)}')
print(f'inner lon : {ex_lon_arr[0]}, {len(ex_lon_arr[0])}')
print(f'inner lon : {ex_lat_arr[0]}, {len(ex_lat_arr[0])}')


# 각 좌표와 중심 좌표 사이의 거리를 계산하는 함수
def calculate_distance_from_center(lon, lat, center_lon, center_lat):
    # 지구 반지름 (대략적인 평균값)
    earth_radius = 6371.0  # km 단위

    # 각도를 라디안으로 변환
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    center_lon_rad, center_lat_rad = np.radians(center_lon), np.radians(center_lat)

    # 경도와 위도의 차이
    delta_lon = lon_rad - center_lon_rad
    delta_lat = lat_rad - center_lat_rad

    # 하버사인 공식 사용
    a = np.sin(delta_lat/2.0)**2 + np.cos(lat_rad) * np.cos(center_lat_rad) * np.sin(delta_lon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # 거리 계산
    distance = earth_radius * c
    return distance

filtered_coords = np.array([
    (lon, lat) for lon, lat in zip(outer_lon_arr, outer_lat_arr)
    if calculate_distance_from_center(lon, lat, center_lon, center_lat) > 200
])

filtered_lon_arr, filtered_lat_arr = filtered_coords[:, 0], filtered_coords[:, 1]

# 결과 출력
print(len(filtered_lon_arr), len(filtered_lat_arr))

"""
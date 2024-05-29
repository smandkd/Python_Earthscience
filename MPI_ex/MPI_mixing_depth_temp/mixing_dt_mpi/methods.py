import numpy as np
import os
import xarray as xr

from netCDF4 import Dataset
import pandas as pd
from MPI_ex.MPI_mixing_depth_temp.mixing_dt_mpi.mixing_depth_temp import mixing_depth

from scipy.spatial import cKDTree
from scipy.interpolate import splrep, splev 

base_salt_dict = r'/home/data/ECCO2/3DAYMEAN/SALT/'
fileEx = r'.nc'
base_opt_dict = r'/home/data/ECCO2/3DAYMEAN/THETA/'

def preprocess_IBTrACS(dataset, sid, agency):
    filtered_dataset = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
    agency_dataset = filtered_dataset.where((filtered_dataset.usa_agency == agency), drop=True)
    tc_dataset = agency_dataset.where(agency_dataset.sid == sid, drop=True)

    wind_array = tc_dataset.usa_wind.data[0]
    gen_index = gen(wind_array)
    lmi_index = lmi(wind_array)
    selected_indices = np.arange(gen_index, lmi_index+1)
    gen2lmi_dataset = tc_dataset.isel(date_time=selected_indices, drop=True)
    
    return gen2lmi_dataset

def lmi(wind_array):
    if np.all(np.isnan(wind_array)):
        return None
    max_value = np.nanmax(wind_array)
    if np.isnan(max_value):
        return None
    
    lmi_index = np.where(wind_array == max_value)[0]
    
    if lmi_index.size > 0:
        return lmi_index[0]
    else:
        return None    

def gen(wind_array):
    gen_index = np.where(wind_array >= 34)[0]
    
    if gen_index.size > 0:
        return gen_index[0]
    else:
        return None    

def isel_time_dataset(data, index):
    result = data.isel(time=index, drop=True)
    return result

def isel_level_dataset(data, index):
    result = data.isel(level=index, drop=True)
    return result

def sel_lat_lon_dataset(data, lat, lon):
    result = data.sel(latitude=lat, longitude=lon, drop=True)
    return result

def TC_pre_3_date(dataset):
    date = dataset['time']
    pre_time = date - pd.Timedelta(days=3)
    
    pre_dt = []
    
    for data in pre_time[0]:
        datetime = np.array(data, dtype='datetime64[ns]')
        time = np.datetime64(datetime, 'D')
        pre_dt.append(time)
    
    return pre_dt

def TC_present_date(dataset):
    date = dataset['time']
    preset_time = date
    
    dt = []
    
    for data in preset_time[0]:
        datetime = np.array(data, dtype='datetime64[ns]')
        time = np.datetime64(datetime, 'D')
        dt.append(time)
        
    return dt

def TC_usa_center_points(dataset):
    lat = np.array(dataset.usa_lat[0], dtype=object)
    lon = np.array(dataset.usa_lon[0], dtype=object)
    center_points = np.column_stack((lon,lat))
    
    return center_points

def sort_level_downgrade(dataset):
    new_arr = dataset.level.data[::-1]
    
    return new_arr

def create_TC_dataframe(dataset):
    haitang_arr = []
    arr_time_len = len(dataset.date_time)
    pre_dt = TC_pre_3_date(dataset)
    
    for i in range(arr_time_len):
        haitang_i = dataset.isel(date_time=i, drop=True)
        
        data = {
            'time': pre_dt[i],
            'usa_wind' : haitang_i.usa_wind.data[0],
            'usa_lat' : haitang_i.usa_lat.data[0],
            'usa_lon' : haitang_i.usa_lon.data[0],
            'usa_r34': haitang_i.usa_r34.data[0],
            'storm_speed': haitang_i.storm_speed.data[0],
            'usa_rmw' : haitang_i.usa_rmw.data[0]
        }
        
        haitang_arr.append(data)
        
    new_dataframe = pd.DataFrame(haitang_arr)
    
    return new_dataframe

def points_tc_netcdf(dataset, tc_coord, tc_thres):
    
    lon = dataset['LONGITUDE_T'][:]
    lat = dataset['LATITUDE_T'][:] 

    coords = np.array(np.meshgrid(lon, lat)).T.reshape(-1, 2)
    
    tree = cKDTree(coords)
    indices = tree.query_ball_point(tc_coord, tc_thres)
    
    points_tc = coords[indices]
    
    new_lon = points_tc[:, 0]
    new_lat = points_tc[:, -1]
    
    return new_lon, new_lat

def find_matching_salt_file(date, salt_path):
    tc_date = date
    salt_year_directory = os.path.join(salt_path, str(tc_date.year))
    
    salt_files = os.listdir(salt_year_directory)
    
    date = str(tc_date).split(' ')[0]
    tc_year = date.split('-')[0]
    tc_month = date.split('-')[1]
    tc_day = date.split('-')[2]
    
    date = tc_year + tc_month + tc_day
    date_1 = tc_year + tc_month + (str(int(tc_day) - 1)).zfill(2)
    date_2 =  tc_year + tc_month + (str(int(tc_day) + 1)).zfill(2)
    
    day_name = 'SALT.1440x720x50.' + date + '.nc'
    day_name_1 = 'SALT.1440x720x50.' + date_1 + '.nc'
    day_name_2 = 'SALT.1440x720x50.' + date_2 + '.nc'    
    
    if day_name in salt_files:
        print(os.path.join(salt_year_directory, day_name))
        return os.path.join(salt_year_directory, day_name)
    elif day_name_1 in salt_files:
        print(os.path.join(salt_year_directory, day_name_1))
        return os.path.join(salt_year_directory, day_name_1)
    else:
        print(os.path.join(salt_year_directory, day_name_2))
        return os.path.join(salt_year_directory, day_name_2)

# 경도 좌표를 새로운 배열로 대체
def replace_longitude(dataset, new_longitude):
    if len(new_longitude) == len(dataset['LONGITUDE_T']):
        dataset['LONGITUDE_T'] = new_longitude
    else:
        raise ValueError("새로운 경도 배열의 크기가 데이터셋의 경도 차원과 일치하지 않습니다.")
    
def open_dataset(date):
    salt_dict = find_matching_salt_file(date, base_salt_dict)
    opt_dict = find_matching_opt_file(date, base_opt_dict)
    
    # 데이터셋 열기
    salt_dataset = xr.open_dataset(salt_dict)
    opt_dataset = xr.open_dataset(opt_dict)
    
     # 새 경도 배열 생성
    new_longitude = np.arange(-179.875, 180.125, 0.25)

    replace_longitude(salt_dataset, new_longitude)
    replace_longitude(opt_dataset, new_longitude)
    
    return salt_dataset, opt_dataset

def open_dataset_netcdf(date):
    salt_dict = find_matching_salt_file(date, base_salt_dict)
    opt_dict = find_matching_opt_file(date, base_opt_dict)
    
    # 데이터셋 열기
    dataset_1 = Dataset(salt_dict, mode='r+')  # 읽기 및 쓰기 모드로 열기
    dataset_2 = Dataset(opt_dict, mode='r+')  # 읽기 및 쓰기 모드로 열기\
    
     # 새 경도 배열 생성
    new_longitude = np.arange(-179.875, 180.125, 0.25)

    replace_longitude(dataset_1, new_longitude)
    replace_longitude(dataset_2, new_longitude)
    
    return dataset_1, dataset_2

def find_matching_opt_file(date, opt_path):
    tc_date = date
    opt_year_directory = os.path.join(opt_path, str(tc_date.year))
    
    opt_files = os.listdir(opt_year_directory)
    
    date = str(tc_date).split(' ')[0]
    tc_year = date.split('-')[0]
    tc_month = date.split('-')[1]
    tc_day = date.split('-')[2]
    
    date = tc_year + tc_month + tc_day
    date_1 = tc_year + tc_month + (str(int(tc_day) - 1)).zfill(2)
    date_2 =  tc_year + tc_month + (str(int(tc_day) + 1)).zfill(2)
    
    day_name = 'THETA.1440x720x50.' + date + '.nc'
    day_name_1 = 'THETA.1440x720x50.' + date_1 + '.nc'
    day_name_2 = 'THETA.1440x720x50.' + date_2 + '.nc'    
    
    if day_name in opt_files:
        return os.path.join(opt_year_directory, day_name)
    elif day_name_1 in opt_files:
        return os.path.join(opt_year_directory, day_name_1)
    else:
        return os.path.join(opt_year_directory, day_name_2)
    
def donut_points(dataset, tc_coord, inner_thres=200/111, outer_thres=800/111):
    dims = dataset.dims
    
    if 'LONGITUDE_T' in dims:
        lon = dataset.LONGITUDE_T.data 
        lat = dataset.LATITUDE_T.data 
    elif 'longitude' in dims :
        lon = dataset.longitude.data 
        lat = dataset.latitude.data
    else:
        lon = dataset.lon.data 
        lat = dataset.lat.data
    
    coords = np.array(np.meshgrid(lon, lat)).T.reshape(-1, 2)
    
    tree = cKDTree(coords)
    
    inner_indices = tree.query_ball_point(tc_coord, inner_thres)
    outer_indices = tree.query_ball_point(tc_coord, outer_thres)
    
    outer_coords = coords[outer_indices]
    inner_coords = set(map(tuple, coords[inner_indices]))
    
    donut_coords = np.array([coord for coord in outer_coords if tuple(coord) not in inner_coords])

    donut_lon = donut_coords[:, 0]
    donut_lat = donut_coords[:, 1]
    
    return donut_lon, donut_lat

def points_tc(dataset, tc_coord, tc_thres):
    dims = dataset.dims
    
    if 'LONGITUDE_T' in dims:
        lon = dataset.LONGITUDE_T.data 
        lat = dataset.LATITUDE_T.data 
    elif 'longitude' in dims :
        lon = dataset.longitude.data 
        lat = dataset.latitude.data
    else:
        lon = dataset.lon.data 
        lat = dataset.lat.data
    
    coords = np.array(np.meshgrid(lon, lat)).T.reshape(-1, 2)
    
    tree = cKDTree(coords)
    indices = tree.query_ball_point(tc_coord, tc_thres)
    
    points_tc = np.array(coords[indices])
    
    new_lon = points_tc[:, 0]
    new_lat = points_tc[:, -1]
    
    return new_lon, new_lat

def sel_lat_lon_dataset(data, lat, lon):
    dims = data.dims
    
    if 'LONGITUDE_T' in dims:
        result = data.sel(LATITUDE_T=lat, LONGITUDE_T=lon, drop=True, method="nearest")
        return result
    elif 'longitude' in dims :
        result = data.sel(latitude=lat, longitude=lon, drop=True, method="nearest")
        return result
    else:
        result = data.sel(lat=lat, lon=lon, drop=True, method="nearest")
        return result
    
def sel_lat_lon_dataset_netcdf(data, lat, lon):
    dims = data.dimensions
    
    if 'LONGITUDE_T' in dims:
        longitude_indices = [np.abs(data['LONGITUDE_T'] - TC_lon).argmin() for TC_lon in lon]
        latitude_indices = [np.abs(data['LATITUDE_T'] - TC_lat).argmin() for TC_lat in lat]

        if 'SALT' in data.variables:
            # salinity 데이터에서 태풍 반경 안의 데이터 추출
            # 데이터셋의 차원에 따라 인덱싱 방식이 다를 수 있습니다
            extracted_salinity = data['SALT'][:, :, latitude_indices, longitude_indices]
        else: 
            extracted_salinity = data['THETA'][:, :, latitude_indices, longitude_indices]
        
        return extracted_salinity
    elif 'longitude' in dims :
        result = data.sel(latitude=lat, longitude=lon, drop=True, method="nearest")
        return result
    else:
        result = data.sel(lat=lat, lon=lon, drop=True, method="nearest")
        return result
    
def calculate_depth_mixing_d_t(data, mix_prof_type):
    count = 0
    Tmix_list = []
    Dmix_list = [] 
    donut_lat = []
    donut_lon = []

    for index, row in data.iterrows():
        t_list = []
        d_list = []
        
        lat = row.usa_lat
        lon = row.usa_lon
        date = row.time
        print(date)
        tc_rad = row.usa_rmw
        
        salt_dataset, opt_dataset = open_dataset(date)
        tc_coord = [lon, lat]
        
        if 'donut' in mix_prof_type:
            tc_rad_dres = 200/111 
            sal_lon, sal_lat = donut_points(salt_dataset, tc_coord, tc_rad_dres)
            opt_lon, opt_lat = donut_points(opt_dataset, tc_coord, tc_rad_dres)
        elif 'usa_rmw' in mix_prof_type:
            tc_rad_dres = (tc_rad * 1.852)/111
            sal_lon, sal_lat = points_tc(salt_dataset, tc_coord, tc_rad_dres)
            opt_lon, opt_lat = points_tc(opt_dataset, tc_coord, tc_rad_dres)
            print('usa_rmw')
            
        donut_lat.append(sal_lat)
        donut_lon.append(sal_lon)
        
        if (len(sal_lon) == 0) or (len(opt_lon) == 0):
            Tmix_list.append(Tmix_list[-1] - 1)
            Dmix_list.append(Dmix_list[-1] - 1)
            break
        
        salt_haitang = sel_lat_lon_dataset(salt_dataset, sal_lat, sal_lon)
        theta_haitang = sel_lat_lon_dataset(opt_dataset, opt_lat, opt_lon)
        salt_data = salt_haitang.SALT.data
        theta_data = theta_haitang.THETA.data
        depth = salt_dataset['DEPTH_T'][:]
        len_lat = len(sal_lat)
        
        D = depth
        TS = row.storm_speed * 0.51444 # knots to m/s
        R = tc_rad * 1852 # nmile to m 
        #R = 200 * 1000 # km to m 
        Vmax = row.usa_wind * 0.51444 # knots to m/s
        
        for i in range(len_lat):
            count += 1
            S = salt_data[:, :, i, i][0]
            T = theta_data[:, :, i, i][0]
            Dmix, Tmix, dens, FT, Tx = mixing_depth(D, T, S, Vmax, TS, R)
            t_list.append(Tmix)                   
            d_list.append(Dmix)      
                    
        Tmix_list.append(np.nanmean(t_list))
        Dmix_list.append(np.nanmean(d_list))
        print(f'{date} mixingT : {Tmix_list} mixingD : {Dmix_list}')
    return Tmix_list, Dmix_list, donut_lon, donut_lat

def shum_mean(dt, tc_coords, pre_dataset, area_type):
    shum_mean_arr = []
    shum_10_list = []
    input_area_lat = []
    input_area_lon = []
    
    for index, time in enumerate(dt):
        shum_arr = []
        tc_coord = tc_coords[index]
        shum_dataset = isel_time_dataset(pre_dataset, index)
        
        if 'donut' in area_type:
            new_lon, new_lat = donut_points(shum_dataset, tc_coord)
        elif '500' in area_type:
            rad = int(area_type)
            tc_rad_dres = rad/111
            new_lon, new_lat = points_tc(shum_dataset, tc_coord, tc_rad_dres)
        
        input_area_lat.append(new_lat)
        input_area_lon.append(new_lon)
        len_level = len(shum_dataset.level.data)
        filtered_shum_dataset = sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
        
        for level in range(len_level):
            dataset = isel_level_dataset(filtered_shum_dataset, level)
            shum = dataset.q.data * 1000
            shum_mean = np.nanmean(shum)
            shum_arr.append(shum_mean)
            
        print(f'{time} : {shum_arr[::-1]}')
        shum_mean_arr.append(shum_arr[::-1])
        
    for i in range(len(shum_mean_arr)):
        shum_10_list.append(shum_mean_arr[i][0])

    print('========================================================')        
    return shum_mean_arr, shum_10_list

def mslp_mean(pre_dt, tc_coords, pre_dataset, area_type):
    mslp_daily_averages = []
    
    for index, time in enumerate(pre_dt):
        mslp_dataset = isel_time_dataset(pre_dataset, index)
        tc_coord = tc_coords[index]
        
        if 'donut' in area_type:
            new_lon, new_lat = donut_points(mslp_dataset, tc_coord)
            print(new_lon)
        elif '500' in area_type:
            rad = int(area_type)
            tc_rad_dres = rad/111
            new_lon, new_lat = points_tc(mslp_dataset, tc_coord, tc_rad_dres)

        filtered_mslp_dataset = sel_lat_lon_dataset(mslp_dataset, new_lat, new_lon)
        
        msl_arr = filtered_mslp_dataset.msl.data[0]/100
        print(f'{time} : {msl_arr}')
        msl_mean = np.mean(msl_arr)
        mslp_daily_averages.append(msl_mean)
        
    return mslp_daily_averages

def airt_mean(dt, tc_coords, pre_dataset, area_type):
    airt_array = []
    airt_10_list = []
    
    for index, time in enumerate(dt):
        level_airt_arr = []
        
        airt_dataset = isel_time_dataset(pre_dataset, index)
        tc_coord = tc_coords[index]
        
        if 'donut' in area_type:
            new_lon, new_lat = donut_points(airt_dataset, tc_coord)
        elif '500' in area_type:
            rad = int(area_type)
            tc_rad_dres = rad/111
            new_lon, new_lat = points_tc(airt_dataset, tc_coord, tc_rad_dres)
            
        new_airt_data = sel_lat_lon_dataset(airt_dataset, new_lat, new_lon)
        level = len(airt_dataset.level.data)
        
        for level in range(level):
            dataset = isel_level_dataset(new_airt_data, level)
            airt_arr = dataset.t.data - 273.15
            airt_mean = np.mean(airt_arr)
            level_airt_arr.append(airt_mean)
        print(f'{time} : {level_airt_arr[::-1]}')    
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')

        airt_array.append(level_airt_arr[::-1])
        
    for i in range(len(airt_array)):
        airt_10_list.append(airt_array[i][0])
        
    return airt_array, airt_10_list
    
def sst_mean(dt, tc_coords, pre_dataset, area_type):
    input_area_lon = []
    input_area_lat = []
    sst_arr = []
    
    for index, time in enumerate(dt):
        oisst_dataset = isel_time_dataset(pre_dataset, index)

        tc_coord = tc_coords[index]
        
        if 'donut' in area_type:
            new_lon, new_lat = donut_points(oisst_dataset, tc_coord)
            print(new_lon)
        elif '500' in area_type:
            rad = int(area_type)
            tc_rad_dres = rad/111
            new_lon, new_lat = points_tc(oisst_dataset, tc_coord, tc_rad_dres)
        
        input_area_lat.append(new_lat)
        input_area_lon.append(new_lon)
        filtered_sst_dataset = sel_lat_lon_dataset(oisst_dataset, new_lat, new_lon)
        
        sst = filtered_sst_dataset.sst.data[0]
        print(f'{time} : {sst}')
        sst_average = np.nanmean(sst) 
        sst_arr.append(sst_average)
    
    return input_area_lon, input_area_lat, sst_arr

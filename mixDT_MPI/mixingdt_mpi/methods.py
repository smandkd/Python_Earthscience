import numpy as np
import os
import xarray as xr

from netCDF4 import Dataset
import pandas as pd
from .mixing_depth_temp import mixing_depth

from scipy.spatial import cKDTree
from scipy.interpolate import splrep, splev 

base_salt_dict = r'/home/data/ECCO2/3DAYMEAN/SALT/'
fileEx = r'.nc'
base_opt_dict = r'/home/data/ECCO2/3DAYMEAN/THETA/'
outer_thres = 800/111

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

def points_tc(dataset, tc_coord, tc_thres):
    
    lon = dataset.LONGITUDE_T.data
    # print(lon)
    lat = dataset.LATITUDE_T.data
    # print(lat)
    coords = np.array(np.meshgrid(lon, lat)).T.reshape(-1, 2)
    
    tree = cKDTree(coords)
    indices = tree.query_ball_point(tc_coord, tc_thres)
    
    points_tc = coords[indices]
    #print(f'boundary coords : {points_tc}')
    
    new_lon = points_tc[:, 0]
    new_lat = points_tc[:, -1]
    
    return new_lon, new_lat

def points_tc_netcdf(dataset, tc_coord, tc_thres):
    
    lon = dataset['LONGITUDE_T'][:]
    # print(lon)
    lat = dataset['LATITUDE_T'][:] 
    # print(lat)
    coords = np.array(np.meshgrid(lon, lat)).T.reshape(-1, 2)
    
    tree = cKDTree(coords)
    indices = tree.query_ball_point(tc_coord, tc_thres)
    
    points_tc = coords[indices]
    #print(f'boundary coords : {points_tc}')
    
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
    
def donut_points(dataset, tc_coord, inner_thres):
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
    # print(f'coords : {coords}')
    
    tree = cKDTree(coords)
    
    inner_indices = tree.query_ball_point(tc_coord, inner_thres)
    outer_indices = tree.query_ball_point(tc_coord, outer_thres)
    
    outer_coords = coords[outer_indices]
    inner_coords = set(map(tuple, coords[inner_indices]))
    ex_coords = coords[inner_indices]
    
    donut_coords = np.array([coord for coord in outer_coords if tuple(coord) not in inner_coords])

    donut_lon = donut_coords[:, 0]
    donut_lat = donut_coords[:, 1]
    
    return donut_lon, donut_lat


def isel_time_dataset(data, index):
    result = data.isel(time=index, drop=True)
    return result

def isel_level_dataset(data, index):
    result = data.isel(level=index, drop=True)
    return result

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
    
def calculate_depth_mixing_d_t(data):
    ecco2_haitang_lon = [] # longitude in TC radius
    ecco2_haitang_lat = [] # latitude in TC radius
    Tmix_list = []
    Dmix_list = [] 
    Theta_list = []
    Salt_list = []
    Dens_list = []
    FT_list = []
    Tx_list = []
    
    for index, row in data.iterrows():
        t_list = []
        d_list = []
        tmix_list = []
        dmix_list = []
        theta_list = []
        the_list = []
        salt_list = []
        sa_list = []
        dens_list = []
        de_list = []
        f_list = []
        ft_list = []
        x_list = []
        tx_list = []
        
        print(f'index : {index}')
        lat = row.usa_lat
        lon = row.usa_lon
        date = row.time
        print(date)
        tc_rad = row.usa_rmw
        tc_rad_dres = (tc_rad * 1.852)/111
        # tc_rad_dres = 200/111 
        salt_dataset, opt_dataset = open_dataset(date)
        tc_coord = [lon, lat]
        
        sal_lon, sal_lat = points_tc(salt_dataset, tc_coord, tc_rad_dres)
        opt_lon, opt_lat = points_tc(opt_dataset, tc_coord, tc_rad_dres)
        
        if (len(sal_lon) == 0) or (len(opt_lon) == 0):
            Tmix_list.append(Tmix_list[-1] - 1)
            Dmix_list.append(Dmix_list[-1] - 1)
            Theta_list.append(Theta_list[-1])
            Salt_list.append(Salt_list[-1])
            Tx_list.append(Tx_list[-1])
            Dens_list.append(Dens_list[-1])
            FT_list.append(FT_list[-1])
            
            break
        
        ecco2_haitang_lon.append(sal_lon)
        ecco2_haitang_lat.append(sal_lat)

        salt_haitang = sel_lat_lon_dataset(salt_dataset, sal_lat, sal_lon)
        theta_haitang = sel_lat_lon_dataset(opt_dataset, opt_lat, opt_lon)
        salt_data = salt_haitang.SALT.data
        theta_data = theta_haitang.THETA.data
        depth = salt_dataset['DEPTH_T'][:]
        len_lat = len(sal_lat)
        len_lon = len(sal_lon)
        
        D = depth
        TS = row.storm_speed * 0.51444 # knots to m/s
        R = tc_rad * 1852 # nmile to m 
        # R = 200 * 1000 # km to m 
        Vmax = row.usa_wind * 0.51444
        
        for i in range(len_lat):
            for j in range(len_lon):
                S = salt_data[:, :, i, j][0]
                T = theta_data[:, :, i, j][0]
                Dmix, Tmix, dens, FT, Tx = mdt.mixing_depth(D, T, S, Vmax, TS, R)
                print(dens) # 0m 깊이 density
                t_list.append(Tmix)                   
                d_list.append(Dmix)         
                de_list.append(np.nanmean(dens[0:30]))
                f_list.append(FT)
                x_list.append(Tx)
                
                the_list.append(np.nanmean(theta_data[:, 0, i, j][0]))
                sa_list.append(np.nanmean(salt_data[:, 0, i, j][0]))
                            
            tmix_list.append(np.nanmean(t_list))
            dmix_list.append(np.nanmean(d_list))
            theta_list.append(np.nanmean(the_list))
            salt_list.append(np.nanmean(sa_list))
            dens_list.append(np.nanmean(de_list))
            ft_list.append(np.nanmean(f_list))
            tx_list.append(np.nanmean(x_list))
            
            t_list = []
            d_list = []
            # print(Tmix)
            # print('**************************************************************')
        Tmix_list.append(np.nanmean(tmix_list))
        Dmix_list.append(np.nanmean(dmix_list))
        Theta_list.append(np.nanmean(theta_list))
        Salt_list.append(np.nanmean(salt_list))
        Dens_list.append(np.nanmean(dens_list))
        FT_list.append(np.nanmean(ft_list))
        Tx_list.append(np.nanmean(tx_list))
        tmix_list = []
        dmix_list = []
        print(Tmix_list)
        
    return Tmix_list, Dmix_list, Theta_list, Salt_list, Dens_list, FT_list, Tx_list

def shum_donut_mean(tc_coords, pre_dataset):
    shum_mean_arr = []
    
    lenght = len(tc_coords)

    for i in range(0, lenght):
        shum_arr = []
        tc_coord = tc_coords[i]
        tc_thres = 200 / 111
        
        shum_dataset = isel_time_dataset(pre_dataset, i)
        new_lon, new_lat = donut_points(shum_dataset, tc_coord, tc_thres)
        
        len_level = len(shum_dataset.level.data)
        filtered_shum_dataset = sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
        
        for level in range(len_level):
            dataset = isel_level_dataset(filtered_shum_dataset, level)
            
            shum = dataset.q.data * 1000
            
            shum_mean = np.nanmean(shum)
            shum_arr.append(shum_mean)
        
        shum_mean_arr.append(shum_arr[::-1])
        print(shum_arr[::-1])
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        
        shum_10_list = []
        
    for i in range(len(shum_mean_arr)):
        print(shum_mean_arr[i][0])
        shum_10_list.append(shum_mean_arr[i][0])
        
    shum_10_list
        
    return shum_mean_arr, shum_10_list

def mslp_donut_mean(pre_dt, tc_coords, pre_dataset):
    mslp_daily_averages = []

    for index, time in enumerate(pre_dt):
        
        mslp_dataset = isel_time_dataset(pre_dataset, index)
        tc_thres = 200 / 111
        tc_coord = tc_coords[index]
        
        new_lon, new_lat = donut_points(mslp_dataset, tc_coord, tc_thres)
        filtered_mslp_dataset = sel_lat_lon_dataset(mslp_dataset, new_lat, new_lon)
        
        msl_arr = filtered_mslp_dataset.msl.data[0]/100
        print(f'{time}  {msl_arr}')
        msl_mean = np.mean(msl_arr)
        mslp_daily_averages.append(msl_mean)
        
    return mslp_daily_averages

def airt_donut_mean(dt, tc_coords, pre_dataset):
    airt_array = []

    for index, time in enumerate(dt):
        level_airt_arr = []
        
        airt_dataset = isel_time_dataset(pre_dataset, index)
        tc_thres = 200 / 111
        tc_coord = tc_coords[index]
        
        new_lon, new_lat = donut_points(airt_dataset, tc_coord, tc_thres)
        new_airt_data = sel_lat_lon_dataset(airt_dataset, new_lat, new_lon)
        level = len(airt_dataset.level.data)
        
        for level in range(level):
            dataset = isel_level_dataset(new_airt_data, level)
            airt_arr = dataset.t.data - 273.15
            print(f'{time} {level} {airt_arr}')
            airt_mean = np.mean(airt_arr)
            
            level_airt_arr.append(airt_mean)
            print('==================================')
            
        airt_array.append(level_airt_arr[::-1])

    airt_10_list = []

    for i in range(len(airt_array)):
        airt_10_list.append(airt_array[i][0])
        
    return airt_array, airt_10_list    
    
    


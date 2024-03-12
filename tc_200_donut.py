#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import seawater as sw
import pandas as pd
import os
import statistics
#%%
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
#%%
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp'), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True) # Haitang 2005
#%%

wind_array = tc_HAITANG.usa_wind.data[0]
gen_index = gen(wind_array)
lmi_index = lmi(wind_array)
selected_indices = np.arange(gen_index, lmi_index+1)
haitang_dataset = tc_HAITANG.isel(date_time=selected_indices, drop=True)

#%%

haitang_time = haitang_dataset['time'].data[0] - pd.Timedelta(days=3)
#%%
haitang_time

# %%
base_salt_dict = r'/home/data/ECCO2/3DAYMEAN/SALT/'
fileEx = r'.nc'
base_opt_dict = r'/home/data/ECCO2/3DAYMEAN/THETA/'
# %%
arr = []
arr_time_len = len(haitang_dataset.date_time)

for i in range(arr_time_len):
    haitang_i = haitang_dataset.isel(date_time=i, drop=True)
    
    data = {
        'time': haitang_time[i],
        'usa_wind' : haitang_i.usa_wind.data[0],
        'usa_lat' : haitang_i.usa_lat.data[0],
        'usa_lon' : haitang_i.usa_lon.data[0],
        'usa_r34': haitang_i.usa_r34.data[0],
        'storm_speed': haitang_i.storm_speed.data[0]
    }
    
    arr.append(data)

# %%
df = pd.DataFrame(arr)
df.head(20)
#%%
all_r34_values = []
for r34_list in df['usa_r34']:
    # 각 리스트에 대해 NaN이 아닌 값만 추가
    all_r34_values.extend([val for val in r34_list if not np.isnan(val)])

# 모든 유효한 'usa_r34' 값의 평균 계산
average_r34 = np.mean(all_r34_values)

#%%
def ra_windstr(u, v):
    """
    
    Wind stress를 계산하는 함수
    
    :param u: Zonal wind component [m/s], 2D array
    :param v: Meridional wind component [m/s], 2D array
    :return: Tx, Ty - Zonal and Meridional wind stress [N/m^2]
    """
    if np.isscalar(u) or u.dim == 1:
        u = np.array([[u]])
    
    if np.isscalar(v) or v.ndim == 1:
        if np.isscalar(v) and v == 0:
            v = np.zeros_like(u)
        else:
            v = np.array([v])
    
    # 입력 배열의 크기 확인
    if u.shape != v.shape:
        raise ValueError('ra_windstr: SIZE of both wind components must be SAME')
    
    # 상수 정의
    roh = 1.2  # 공기 밀도, kg/m^3
    
    # Wind Stresses 계산
    lt, ln = u.shape
    Tx = np.full((lt, ln), np.nan)
    Ty = np.full((lt, ln), np.nan)
    for ii in range(lt):
        for jj in range(ln):
            U = np.sqrt(u[ii, jj]**2 + v[ii, jj]**2)  # Wind speed
            # Cd 계산
            if U <= 1:
                Cd = 0.00218
            elif U > 1 and U <= 3:
                Cd = (0.62 + 1.56 / U) * 0.001
            elif U > 3 and U < 10:
                Cd = 0.00114
            else:
                Cd = (0.49 + 0.065 * U) * 0.001
            
            # Tx, Ty 계산
            Tx[ii, jj] = Cd * roh * U * u[ii, jj]
            Ty[ii, jj] = Cd * roh * U * v[ii, jj]
    
    return Tx, Ty
# %%
def mixing_depth(D, T, S, Vmax, TS, R):
    valid_indices = ~np.isnan(T) & ~np.isnan(S) & ~np.isnan(D)
    D = D[valid_indices]
    T = T[valid_indices]
    S = S[valid_indices]
    
    Tx, Ty = ra_windstr(Vmax, 0) 
    # caculate wind stress, 해양 표층에 작용하는 힘을 나타내며, 수직 혼합에 영향을 준다 
    
    max_depth = np.floor(np.max(D)) + 1
    depth_range = np.arange(0, max_depth + 1, 1)

    FT = 4*(R/TS) # residence time(s)
    p0 = 1024 # sea water density (kg/m3)
    SN = 1.2 # S number - nondimensional storm speed

    interp_T = interp1d(D, T, kind='linear', fill_value='extrapolate')
    interp_S = interp1d(D, S, kind='linear', fill_value='extrapolate')
    
    T1 = interp_T(depth_range)
    S1 = interp_S(depth_range)
    
    T1[:5] = T1[5]
    S1[:5] = S1[5]
    
    Dmix = np.zeros(1)
    Tmix = np.zeros(1)
    
    dens = sw.dens(S1 ,T1, p0)
    
    for d in range(int(max_depth)):
        if d != 0:
            if (9.8 * (d) * (dens[d] - np.mean(dens[:d]))) / p0 * ((SN * (Tx[0][0]/(p0 * d)) * FT) ** 2) >= 0.6:
                break
        
    Dmix = d + 1 # Mixing depth (m)
    Tmix = np.nanmean(T1[:d + 1]) # Depth-averaged Temperature (degreeC)
    
    return Dmix, Tmix  
# %%
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
    
dis_thres = 200/111
tc_rad_dres = average_r34 / 111

def open_dataset(date):
    salt_dict = find_matching_salt_file(date, base_salt_dict)
    opt_dict = find_matching_opt_file(date, base_opt_dict)
    
    salt_dataset = xr.open_dataset(salt_dict)
    opt_dataset = xr.open_dataset(opt_dict)
    
    salt_dataset['LONGITUDE_T'] = salt_dataset['LONGITUDE_T'] - 180
    opt_dataset['LONGITUDE_T'] = opt_dataset['LONGITUDE_T'] - 180
    
    return salt_dataset, opt_dataset

def donut_points(dataset, coord, radius):
    lon = dataset.LONGITUDE_T.data 
    lat = dataset.LATITUDE_T.data 
    coords = np.array(
        np.meshgrid(lon, lat)
    ).T.reshape(-1, 2)
    tree = cKDTree(coords)
    
    indices_200 = tree.query_ball_point(coord, dis_thres)
    indices_radi = tree.query_ball_point(coord, radius)
    
    coords_200 = coords[indices_200]
    coords_radi = coords[indices_radi]
    
    donut_coords = [tuple(row) for row in coords_200 if tuple(row) not in coords_radi]
    donut_lon = [coord[0] for coord in donut_coords]
    donut_lat = [coord[-1] for coord in donut_coords]
    
    print(donut_lon)
    return donut_lon, donut_lat
    
#%%
def interpolation(salt_dataset, opt_dataset, coord, tc_rad):
    sal_donut_lon, sal_donut_lat = donut_points(salt_dataset, coord, tc_rad)
    
    sal_points = salt_dataset.sel(
        LATITUDE_T=sal_donut_lat,
        LONGITUDE_T=sal_donut_lon, 
        drop=True, method="nearest" 
        )

    sal_depth = sal_points['DEPTH_T']
    new_sal_depth = np.arange(5, sal_depth[-1] + 1, 1)
    interpolated_sal = sal_points.interp(DEPTH_T=new_sal_depth, method="linear")

    opt_donut_lon, opt_donut_lat = donut_points(opt_dataset, coord, tc_rad)
    
    opt_points = opt_dataset.sel(
        LATITUDE_T=opt_donut_lat,
        LONGITUDE_T=opt_donut_lon,
        drop=True, method="nearest"
    )

    opt_depth = opt_points['DEPTH_T']
    new_opt_depth = np.arange(5, opt_depth[-1] + 1, 1)
    interpolated_opt = opt_points.interp(DEPTH_T=new_opt_depth, method="linear")
    
    return interpolated_sal, interpolated_opt, new_sal_depth
#%%
arr_1 =[]
#%%

for index, row in df.iterrows():
    list_1 = []
    lat = row.usa_lat
    lon = row.usa_lon
    date = row.time
    radius = row.usa_r34
    
    salt_dataset, opt_dataset = open_dataset(date)
        
    tc_coord = np.array(
        np.meshgrid(lon, lat)
    ).T.reshape(-1, 2)
    
    inter_sal, inter_opt, D = interpolation(salt_dataset, opt_dataset, tc_coord[0], tc_rad_dres)
    TS = row.storm_speed * 0.514444    
    
    if np.all(np.isnan(row.usa_r34)):
        R = average_r34 * 1852
    else:
        R = np.nanmean(row.usa_r34) * 1852
    
    if np.isnan(R):
        R = 0
    else :
        R = R * 1852
        
    Vmax = row.usa_wind * 0.514444    
    
    for i in range(len(inter_opt.LATITUDE_T)):
        S = inter_sal.SALT.data[0][:, i, i].flatten()
        T = inter_opt.THETA.data[0][:, i, i].flatten()
        
        Dmix, Tmix = mixing_depth(D, T, S, Vmax, TS, R)
        
        list_1.append(Tmix)           
    
    # mean_1 = statistics.mean(list_1)
    # print(f'{tc_coord} {mean_1}')
    
    # arr_1.append(mean_1)
    
#%%

    
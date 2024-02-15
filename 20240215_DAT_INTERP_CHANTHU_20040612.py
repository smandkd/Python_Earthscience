"""
CHANTHU 20040612 센터 포지션으로부터 200km 반경내의 포인트들에 대한 Dmix, Tmix 값 구하기 
"""

#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import seawater as sw
import math
# %%
opt_dataset = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/THETA/2004/THETA.1440x720x50.20040612.nc')
sal_dataset = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/SALT/2004/SALT.1440x720x50.20040612.nc')
#%%
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
new_dataset = dataset.where((dataset.season >= 2004) &
                              (dataset.season <= 2020) & 
                              (dataset.usa_agency == b'jtwc_wp') &
                              (dataset.usa_wind >= 34), drop=True)

tc_CHANTHU = new_dataset.where(new_dataset.sid == b'2004158N07142', drop=True)
indexes = np.where(tc_CHANTHU.usa_wind >= 34)
gen_INDEX = indexes[0][0] # 34
lmi_INDEX = np.where(tc_CHANTHU.usa_wind == tc_CHANTHU.usa_wind.max())[-1][0]
selected_data = tc_CHANTHU.isel(date_time=slice(gen_INDEX, lmi_INDEX+1), drop=True)
chanthu_20040612 = selected_data.isel(date_time=12)

tc_center_coord = np.array(
    np.meshgrid(chanthu_20040612.lat.data, chanthu_20040612.lon.data)
).T.reshape(-1, 2)

dis_thres = 200/111

# %%
opt_dataset['LONGITUDE_T'] = opt_dataset['LONGITUDE_T'] - 180

opt_lon = opt_dataset.LONGITUDE_T.data
opt_lat = opt_dataset.LATITUDE_T.data 

opt_coords = np.array(
    np.meshgrid(opt_lat, opt_lon)
).T.reshape(-1, 2)
opt_tree = cKDTree(opt_coords)
opt_indices = opt_tree.query_ball_point(tc_center_coord[0], dis_thres)
opt_tc_center_nearby_coords = opt_coords[opt_indices]
# %%
sal_lon = sal_dataset.LONGITUDE_T.data 
sal_lat = sal_dataset.LATITUDE_T.data 
sal_coords = np.array(
    np.meshgrid(sal_lat, sal_lon)
).T.reshape(-1, 2)
sal_tree = cKDTree(sal_coords)

sal_indices = sal_tree.query_ball_point(tc_center_coord[0], dis_thres)
sal_tc_center_nearby_coords = sal_coords[sal_indices]

# %%
sal_tc_center_nearby_coords[:, 1]
sal_tc_center_nearby_coords[:, 0]

opt_tc_center_nearby_coords[:, 1]
opt_tc_center_nearby_coords[:, 0]
# %%
sal_points_chanthu_040612 = sal_dataset.sel(
        LATITUDE_T=sal_tc_center_nearby_coords[:, 0],
        LONGITUDE_T=sal_tc_center_nearby_coords[:, 1], 
        drop=True, method="nearest"
        )

opt_points_chanthu_040612 = opt_dataset.sel(
        LATITUDE_T=opt_tc_center_nearby_coords[:, 0],
        LONGITUDE_T=opt_tc_center_nearby_coords[:, 1], 
        drop=True, method="nearest"
        )

# %%
opt_depth = opt_points_chanthu_040612['DEPTH_T']
new_opt_depth = np.arange(5, opt_depth[-1] + 1, 1)
interpolated_opt = opt_points_chanthu_040612.interp(DEPTH_T=new_opt_depth, method="linear")

sal_depth = sal_points_chanthu_040612['DEPTH_T']
new_sal_depth = np.arange(5, sal_depth[-1] + 1, 1)
interpolated_sal = sal_points_chanthu_040612.interp(DEPTH_T=new_sal_depth, method="linear")

# %%
# Salinity, THETA 첫 위경도만 
S_1 = interpolated_sal.SALT.data[0][:, 0, 0].flatten()
"""
Salinity
index : 0 
array([33.47647095, 33.47656898, 33.47666702, ...,         nan,
               nan,         nan])
lon : 110.6
lat : 12.38
depth : 5 ~ 5.907e+03

"""
T_1 = interpolated_opt.THETA[0][:, 0 , 0].flatten()
"""
Ocean potential temperature
index : 0
array([30.20540428, 30.20427914, 30.20315399, ...,         nan,
               nan,         nan])
lon : 110.6
lat : 12.38
depth : 5 ~ 5907

"""
D = interpolated_opt.DEPTH_T.data
"""
Depth , intervals 1m
array([5.000e+00, 6.000e+00, 7.000e+00, ..., 5.905e+03, 5.906e+03,
       5.907e+03])

"""
TS = chanthu_20040612.storm_speed*0.514444
"""
TC translation speed (m/s)
"""
R = chanthu_20040612.usa_r34.mean() * 1852
"""
radius of wind (m)
"""
Vmax_1 = chanthu_20040612.usa_wind.data[0] * 0.514444
# %%
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
Dmix, Tmix = mixing_depth(D, T_1, S_1, Vmax_1, TS, R)
print(f'Dmix, Tmix : {Dmix}, {Tmix}')
# %%
# 태풍 CHANTHU 20040612의 200km 반경 포인트 전부 
lat_size = len(interpolated_opt.LATITUDE_T)
lon_size = len(interpolated_opt.LONGITUDE_T)

D = interpolated_opt.DEPTH_T.data
TS = chanthu_20040612.storm_speed*0.514444
R = chanthu_20040612.usa_r34.mean() * 1852
Vmax_1 = chanthu_20040612.usa_wind.data[0] * 0.514444

for i in range(lat_size):
    for j in range(lon_size):
        S = interpolated_sal.SALT.data[0][:, i, j].flatten()
        T = interpolated_opt.THETA.data[0][:, i, j].flatten()
        
        Dmix, Tmix = mixing_depth(D, T, S, Vmax_1, TS, R)
        print(f'[lat, lon : {i}, {j}]\nDmix, Tmix : {Dmix}, {Tmix}\n\n')

#%%
import xarray as xr
import numpy as np
import seawater as sw
import scipy as scipy
from scipy.interpolate import interp1d

#%%
# 2004-06-12 Salinity Dataset
sal_dataset = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/SALT/2004/SALT.1440x720x50.20040612.nc')
sal_dataset.variables
# %%
# 2004-06-12 Ocean Potential temperature Dataset 
opt_dataset = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/THETA/2004/THETA.1440x720x50.20040612.nc')
opt_dataset

# %%
# 
dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
new_dataset = dataset.where((dataset.season >= 2004) &
                              (dataset.season <= 2020) & 
                              (dataset.usa_agency == b'jtwc_wp') &
                              (dataset.usa_wind >= 34), drop=True)

tc_CHANTHU = new_dataset.where(new_dataset.sid == b'2004158N07142', drop=True) # CHANTHU
# %%
indexes = np.where(tc_CHANTHU.usa_wind >= 34)
gen_INDEX = indexes[0][0] # 34
lmi_INDEX = np.where(tc_CHANTHU.usa_wind == tc_CHANTHU.usa_wind.max())[-1][0]
lon = tc_CHANTHU.usa_lon.data[0][gen_INDEX:lmi_INDEX+1]
lat = tc_CHANTHU.usa_lat.data[0][gen_INDEX:lmi_INDEX+1]
time = tc_CHANTHU.time.data[0][gen_INDEX:lmi_INDEX+1]
usa_wind = tc_CHANTHU.usa_wind.data[gen_INDEX:lmi_INDEX+1] # knots
usa_wind_ms = usa_wind * 0.514444 # knots to m/s

print(f'Chanthu lon, lat : {lon} \n{lat}')
print(f'size of lon, lat : {len(lon)} {len(lat)}')
print(f'Chanthu time : {time}')
print(f'Chanthu wind speed, size : {usa_wind} {len(usa_wind[0])}')
"""
Chanthu lon, lat : [125.4 124.1 122.7 121.3 120.2 119.1 118.2 117.2 116.2 115.2 114.1 112.9
 111.6 110.2] 
[10.6 10.9 11.  11.1 11.4 11.6 12.1 12.4 12.7 13.1 13.3 13.6 13.9 14. ]
size of lon, lat : 14 14
Chanthu time : ['2004-06-09T00:00:00.000040448' '2004-06-09T06:00:00.000040448'
 '2004-06-09T12:00:00.000040448' '2004-06-09T18:00:00.000040448'
 '2004-06-10T00:00:00.000040448' '2004-06-10T06:00:00.000040448'
 '2004-06-10T12:00:00.000040448' '2004-06-10T18:00:00.000040448'
 '2004-06-11T00:00:00.000040448' '2004-06-11T06:00:00.000040448'
 '2004-06-11T12:00:00.000040448' '2004-06-11T18:00:00.000040448'
 '2004-06-12T00:00:00.000040448' '2004-06-12T06:00:00.000040448']
Chanthu wind speed, size : [[35. 35. 40. 45. 45. 45. 45. 45. 45. 50. 55. 60. 65. 75. 65. 50. 35.]] 17

"""
#%%


# %% ra_windstr.m

Vmax = usa_wind_ms.data

def ra_windstr(u, v):
    """
    Wind stress를 계산하는 함수
    
    :param u: Zonal wind component [m/s], 2D array
    :param v: Meridional wind component [m/s], 2D array
    :return: Tx, Ty - Zonal and Meridional wind stress [N/m^2]
    """
    if np.isscalar(u) or u.dim == 1:
        u = np.array([[u]])
    
    if np.isscalar(v) and v.ndim == 1:
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

Tx, Ty = ra_windstr(Vmax, 0)
print(f'Tx : {len(Tx[0])}, {Tx[0]} \nTy : {len(Ty[0])} {Ty[0]}')   

"""
Tx : 17, [0.64594534 0.64594534 0.92864064 1.28283472 1.28283472 1.28283472
 1.28283472 1.28283472 1.28283472 1.71649177 2.23757674 2.85405434
 3.57388871 5.3554893  3.57388871 1.71649177 0.64594534] 
Ty : 17 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
"""
# %%
selected_data = tc_CHANTHU.isel(date_time=slice(gen_INDEX, lmi_INDEX+1), drop=True)
"""
['2004-06-09T00:00:00.000040448', '2004-06-09T06:00:00.000040448',
        '2004-06-09T12:00:00.000040448', '2004-06-09T18:00:00.000040448',
        '2004-06-10T00:00:00.000040448', '2004-06-10T06:00:00.000040448',
        '2004-06-10T12:00:00.000040448', '2004-06-10T18:00:00.000040448',
        '2004-06-11T00:00:00.000040448', '2004-06-11T06:00:00.000040448',
        '2004-06-11T12:00:00.000040448', '2004-06-11T18:00:00.000040448',
        '2004-06-12T00:00:00.000040448', '2004-06-12T06:00:00.000040448']]

"""
selected_data
# %%
selected_data.usa_wind
"""
array([[35., 35., 40., 45., 45., 45., 45., 45., 45., 50., 55., 60., 65.,
        75.]], dtype=float32)
"""
# %%
selected_data.dims
selected_data.isel(date_time=12)
# %%
# 2004-06-11T18:00:00.000040448
storm_040611 = selected_data.isel(date_time=11, drop=True)
new_sal_data = sal_dataset.sel(LATITUDE_T = storm_040611.lat.data, LONGITUDE_T = storm_040611.lon.data, method="nearest", drop=True)
new_opt_data = opt_dataset.sel(LATITUDE_T = storm_040611.lat.data, LONGITUDE_T = storm_040611.lon.data, method="nearest", drop=True)

D = new_sal_data.DEPTH_T.data # depth information 
S = new_sal_data.SALT.data.flatten() # PSU
T = new_opt_data.THETA.data.flatten() # degree C

print(f'depth information : {D} \nsalinity information : {S}\ntheta information : {T}')

"""
depth information : 
[5.000000e+00 1.500000e+01 2.500000e+01 3.500000e+01 4.500000e+01
 5.500000e+01 6.500000e+01 7.500500e+01 8.502500e+01 9.509500e+01
 1.053100e+02 1.158700e+02 1.271500e+02 1.397400e+02 1.544700e+02
 1.724000e+02 1.947350e+02 2.227100e+02 2.574700e+02 2.999300e+02
 3.506800e+02 4.099300e+02 4.774700e+02 5.527100e+02 6.347350e+02
 7.224000e+02 8.144700e+02 9.097400e+02 1.007155e+03 1.105905e+03
 1.205535e+03 1.306205e+03 1.409150e+03 1.517095e+03 1.634175e+03
 1.765135e+03 1.914150e+03 2.084035e+03 2.276225e+03 2.491250e+03
 2.729250e+03 2.990250e+03 3.274250e+03 3.581250e+03 3.911250e+03
 4.264250e+03 4.640250e+03 5.039250e+03 5.461250e+03 5.906250e+03] 
 
salinity information : 
[33.89938  33.906384 34.010506 34.18278  34.31616  34.452797 34.58723
 34.690258 34.757084 34.796158 34.812878 34.809185 34.78884  34.75545
 34.710197 34.65302  34.582874 34.50472  34.42873  34.363647 34.33655
 34.3448   34.36624  34.412838 34.449223 34.476994 34.50483  34.530193
 34.553814 34.570038 34.58271  34.591972 34.59741  34.601307 34.60396
 34.60666  34.6097   34.61274        nan       nan       nan       nan
       nan       nan       nan       nan       nan       nan       nan
       nan]
       
theta information : 
[29.634066  29.6095    28.67262   26.560633  25.398905  24.751732
 24.136414  23.41044   22.564104  21.67885   20.817755  19.953459
 19.044119  18.085236  17.088772  16.039215  14.865415  13.564115
 12.247313  10.810428   9.464983   8.500541   7.833441   7.1761355
  6.4001284  5.78437    5.258677   4.7516727  4.2416286  3.7914984
  3.4027333  3.076465   2.8301024  2.6583202  2.525422   2.422988
  2.35008    2.3041668        nan        nan        nan        nan
        nan        nan        nan        nan        nan        nan
        nan        nan]
"""

# %% 
# date_time : 2004-06-11T18:00:00.000040448

TS = storm_040611.storm_speed.data * 0.514444 
R  = storm_040611.usa_r34.data[0][1] * 1852 
Vmax = storm_040611.usa_wind.data * 0.514444

print(f'TS : {TS} \nVmax : {Vmax} \nR:{R}')
# %% ra_windstr.m

Vmax = usa_wind_ms.data

def ra_windstr(u, v):
    """
    Wind stress를 계산하는 함수
    
    :param u: Zonal wind component [m/s], 2D array
    :param v: Meridional wind component [m/s], 2D array
    :return: Tx, Ty - Zonal and Meridional wind stress [N/m^2]
    """
    if np.isscalar(u) or u.ndim == 1:
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
            
    print(f'Tx : {len(Tx[0])}, {Tx[0]} \nTy : {len(Ty[0])} {Ty[0]}')       
    return Tx, Ty

# %% mixing_depth0(D, T, S, Vmax, TS, R)

def mixing_depth(D, T, S, Vmax, TS, R):
    Tx, Ty = ra_windstr(Vmax, 0)  # caculate wind stress, 해양 표층에 작용하는 힘을 나타내며, 수직 혼합에 영향을 준다 
    print(f'Tx : {Tx}')
    
    max_depth = np.floor(np.max(D))
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
    
    max_depth = np.floor(np.max(D)) + 1
    Dmix = np.zeros(1)
    Tmix = np.zeros(1)
    
    dens = sw.dens(S1 ,T1, p0)
    
    for d in range(int(max_depth)):
        # print(9.8 * (d) * (dens[d] - np.mean(dens[:d])))
        if (9.8 * (d) * (dens[d] - np.mean(dens[:d]))) / p0 * ((SN * (Tx/(p0 * d)) * FT) ** 2) >= 0.6:
            break
        
    Dmix[0] = d + 1 # Mixing depth (m)
    Tmix[0] = np.nanmean(T1[:d + 1]) # Depth-averaged Temperature (degreeC)
    
    return Dmix, Tmix 
        
    

# %%
D = new_sal_data.DEPTH_T.data # depth information 
S = new_sal_data.SALT.data.flatten() # PSU
T = new_opt_data.THETA.data.flatten() # degree C
TS = storm_040611.storm_speed.data * 0.514444 
R  = storm_040611.usa_r34.data[0][1] * 1852 
Vmax = storm_040611.usa_wind.data[0] * 0.514444
# print(f'TS : {TS} \nVmax : {Vmax} \nR:{R}\nD : {D}\nS : {S}\nT : {T}')

a, b = mixing_depth(D, T, S, Vmax, TS, R)
print(f'minxing_depth results : {a} {b}')

# %%







#%%
import xarray as xr
import numpy as np
import seawater as sw
#%%
# 2004-06-12 salinity
sal_dataset = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/SALT/2004/SALT.1440x720x50.20040612.nc')
sal_dataset
#%%
# 2004-06-12 ocean potential temperature
opt_dataset = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/THETA/2004/THETA.1440x720x50.20040612.nc')
# 2004-06-05 ~ 2004-06-15 CHANTHU
dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
new_dataset = dataset.where((dataset.season >= 2004) &
                              (dataset.season <= 2020) & 
                              (dataset.usa_agency == b'jtwc_wp') &
                              (dataset.usa_wind >= 34), drop=True)

tc_CHANTHU = new_dataset.where(new_dataset.sid == b'2004158N07142', drop=True) # CHANTHU

#%%
indexes = np.where(tc_CHANTHU.usa_wind >= 34)
gen_INDEX = indexes[0][0] # 34
lmi_INDEX = np.where(tc_CHANTHU.usa_wind == tc_CHANTHU.usa_wind.max())[-1][0]
lon = tc_CHANTHU.usa_lon.data[0][gen_INDEX:lmi_INDEX+1]
lat = tc_CHANTHU.usa_lat.data[0][gen_INDEX:lmi_INDEX+1]
time = tc_CHANTHU.time.data[0][gen_INDEX:lmi_INDEX+1]
usa_wind = tc_CHANTHU.usa_wind.data[gen_INDEX:lmi_INDEX+1]
print(f'{lon} \n{lat}')
print(f'{len(lon)} \n{len(lat)}')
print(f'{time}')
print(f'{usa_wind}')
#%%
chanthu_sal = sal_dataset.sel(LATITUDE_T= lat, LONGITUDE_T= lon,  method="nearest", drop=True)
chanthu_opt = opt_dataset.sel(LATITUDE_T = lat, LONGITUDE_T = lon, method="nearest", drop=True)
#%%
chanthu_opt
# %%

D = chanthu_sal.DEPTH_T.data
S = chanthu_sal.SALT.data
T = chanthu_opt.THETA.data


# %%

print(len(chanthu_sal.LONGITUDE_T)) # 14
print(len(chanthu_sal.LATITUDE_T)) # 14
print(len(D)) # 50
print(len(S[0])) # 50
print(len(S[0][0])) # 14
print(T)
"""
T : 
[[[[30.181091         nan        nan ... 28.653004  28.92623
    29.094522 ]
   [30.015003         nan        nan ... 28.73076   28.955793
    29.190228 ]
   [       nan        nan        nan ... 28.783813  28.933277
    29.329817 ]
   ...
   [29.39525          nan        nan ... 29.634066  29.786358
    29.939724 ]
   [29.290922         nan        nan ... 29.667463  29.49086
    29.699566 ]
   [29.252369  29.147398         nan ... 29.720022  29.526836
    29.370762 ]]

  [[       nan        nan        nan ... 28.64044   28.915087
    29.081524 ]
   [       nan        nan        nan ... 28.717592  28.94353
    29.17461  ]
   [       nan        nan        nan ... 28.7701    28.919697
    29.312498 ]
...
   [       nan        nan        nan ...        nan        nan
           nan]
   [ 1.2478713        nan        nan ...        nan        nan
           nan]]]]

S : 
[[[[33.60575        nan       nan ... 33.74112  33.712147 33.69934 ]
   [33.66259        nan       nan ... 33.738373 33.667526 33.68064 ]
   [      nan       nan       nan ... 33.74538  33.697598 33.65251 ]
   ...
   [33.985          nan       nan ... 33.89938  33.717373 33.627964]
   [33.963364       nan       nan ... 33.828396 33.861378 33.762505]
   [33.964622 33.85693        nan ... 33.73753  33.846844 33.92613 ]]

  [[      nan       nan       nan ... 33.74382  33.7164   33.706448]
   [      nan       nan       nan ... 33.74118  33.6719   33.68806 ]
   [      nan       nan       nan ... 33.748875 33.701473 33.659847]
   ...
   [33.985497       nan       nan ... 33.906384 33.725277 33.630238]
   [33.965366       nan       nan ... 33.841145 33.87552  33.76941 ]
   [33.967075       nan       nan ... 33.749374 33.86604  33.933212]]

  [[      nan       nan       nan ... 33.94484  33.79933  33.872936]
   [      nan       nan       nan ... 33.94334  33.772537 33.84929 ]
   [      nan       nan       nan ... 33.902718 33.797684 33.79452 ]
   ...
   [33.98632        nan       nan ... 34.010506 33.856293 33.66347 ]
   [33.96809        nan       nan ... 33.95778  33.999245 33.8521  ]
   [33.96929        nan       nan ... 33.872055 33.986156 34.010757]]

  ...
...
   ...
   [      nan       nan       nan ...       nan       nan       nan]
   [      nan       nan       nan ...       nan       nan       nan]
   [34.678566       nan       nan ...       nan       nan       nan]]]]
"""

# %%
# ra_windstr
Vmax = usa_wind.data

def ra_windstr(u, v):
    """
    Wind stress를 계산하는 함수
    
    :param u: Zonal wind component [m/s], 2D array
    :param v: Meridional wind component [m/s], 2D array
    :return: Tx, Ty - Zonal and Meridional wind stress [N/m^2]
    """
    if np.isscalar(v) and v == 0:
        v = np.zeros_like(u)
    
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
    
    print(f'{Tx} {Ty}')
    return Tx, Ty

Tx, Ty = ra_windstr(Vmax, 0)
print(len(Tx[0]))   

# %%
# mixing_depth0.m
"""
Input
D : depth information of T-S profiles
T : Temperature profile
S : Salinity profile
Vmax : maximum wind speed(m/s)
TS : TC translation speed(m/s)
R : radius of wind(m)

Output
Tmix : Depth-averaged Temperature (degreeC)
Dmix : Mixing depth(m)

D = depth

"""

def mixing_depth0(D, T, S, Vmax, TS, R):
    
    return 0
#%%
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
# %%
def znot_m_v8(uref):
    p13 = -1.296521881682694e-02
    p12 =  2.855780863283819e-01
    p11 = -1.597898515251717e+00
    p10 = -8.396975715683501e+00

    p25 =  3.790846746036765e-10
    p24 =  3.281964357650687e-09
    p23 =  1.962282433562894e-07
    p22 = -1.240239171056262e-06
    p21 =  1.739759082358234e-07
    p20 =  2.147264020369413e-05

    p35 =  1.840430200185075e-07
    p34 = -2.793849676757154e-05
    p33 =  1.735308193700643e-03
    p32 = -6.139315534216305e-02
    p31 =  1.255457892775006e+00
    p30 = -1.663993561652530e+01

    p45 =  3.806734376022113e-11
    p44 = -1.138431432537254e-08
    p43 =  1.330088992305179e-06
    p42 = -7.499441958611604e-05
    p41 =  0.001980799375510e+00
    p40 = -0.016943180834872e+00

    if 0.0 <= uref <= 6.5:
        znotm = np.exp(p10 + p11*uref + p12*uref**2 + p13*uref**3)
    elif 6.5 < uref <= 15.7:
        znotm = p25*uref**5 + p24*uref**4 + p23*uref**3 + p22*uref**2 + p21*uref + p20
    elif 15.7 < uref <= 46.0:
        znotm = np.exp(p35*uref**5 + p34*uref**4 + p33*uref**3 + p32*uref**2 + p31*uref + p30)
    elif uref > 46.0:
        znotm = 10 * (1/np.exp(np.sqrt(0.16 / (p45*uref**5 + p44*uref**4 + p43*uref**3 + p42*uref**2 + p41*uref + p40))))
    else:
        raise ValueError(f'Wrong input uref value: {uref}')

    return znotm

#%%

def new_ra_windstr(u):
    roh = 1.2  # 공기 밀도, kg/m^3
    z0 = znot_m_v8(u)
    U = np.sqrt(u**2)  # Wind speed
  
    Cd = (2.5 * np.log(10/z0))**-2
        
    Tx = Cd * U * roh * u
    
    return Tx, Cd

#%%
def ra_windstr(u, v):
    if np.isscalar(u) or u.dim == 1:
        u = np.array([[u]])
    
    if np.isscalar(v) or v.ndim == 1:
        if np.isscalar(v) and v == 0:
            v = np.zeros_like(u)
        else:
            v = np.array([v])
    
    if u.shape != v.shape:
        raise ValueError('ra_windstr: SIZE of both wind components must be SAME')
    
    roh = 1.2  # 공기 밀도, kg/m^3
    
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
    
    return Tx, Cd


# %%
wind_arr = np.arange(1, 70, 1)
#%%
# ====================================
# z0
# ====================================
z0_arr = []
i = 0
while i < len(wind_arr):
    z0 = znot_m_v8(wind_arr[i])
    z0_arr.append(z0)
    i += 1

# ===================================
#    Graph of x:wind, y:wind stress
# ===================================
plt.figure(figsize=(10, 6))
plt.plot(wind_arr, z0_arr, color='green')
plt.xlabel('wind')
plt.ylabel('z0')
plt.title('z0')
plt.show()
#%%

# ====================================
# new wind stress, Cd, graph
# ====================================
new_ws_arr = []
new_cd_arr = []
i = 0
while i < len(wind_arr):
    Tx, Cd= new_ra_windstr(wind_arr[i])
    new_ws_arr.append(Tx)
    new_cd_arr.append(Cd)
    i += 1
    
plt.figure(figsize=(10, 6))
plt.plot(wind_arr, new_ws_arr, color='green')
plt.ylim(0, 25)
plt.xlabel('wind')
plt.ylabel('wind stress')
plt.title('New wind stress')
plt.show()
#%%
plt.figure(figsize=(10, 6))
plt.plot(wind_arr, new_cd_arr, color='green')
plt.ylim(0, 0.0060)
plt.xlabel('wind')
plt.ylabel('Cd')
plt.title('New Cd')
plt.show()
# %%
# =========================================
#  old wind stress, Cd, graph
# =========================================
old_ws_arr = []
old_cd_arr = []

i = 0
while i < len(wind_arr):
    Tx, Cd = ra_windstr(wind_arr[i], 0)
    old_ws_arr.append(Tx)
    old_cd_arr.append(Cd)
    i += 1

old_ws_arr_2 = [item[0][0] for item in old_ws_arr]
old_ws_arr_2

plt.figure(figsize=(10, 6))
plt.plot(wind_arr, old_ws_arr_2, color='green')
plt.xlabel('wind')
plt.ylabel('wind stress')
plt.title('Old wind stress')
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.plot(wind_arr, old_cd_arr, color='green')
plt.xlabel('wind')
plt.ylabel('Cd')
plt.title('Old Cd')
plt.ylim(0, 0.0060)
plt.show()
# %%
# =============================================
# Old Cd, New Cd 동시에 그래프
# =============================================

plt.plot(wind_arr, old_cd_arr, color='green', label='old Cd')
plt.plot(wind_arr, new_cd_arr, color='deeppink', label='new Cd')
plt.title('Cd')
plt.legend(loc='upper left')
plt.show()

# %%

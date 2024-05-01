import numpy as np
import seawater as sw
from scipy.interpolate import interp1d

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

def new_ra_windstr(u):
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

    if 0.0 <= u <= 6.5:
        znotm = np.exp(p10 + p11*u + p12*u**2 + p13*u**3)
    elif 6.5 < u <= 15.7:
        znotm = p25*u**5 + p24*u**4 + p23*u**3 + p22*u**2 + p21*u + p20
    elif 15.7 < u <= 46.0:
        znotm = np.exp(p35*u**5 + p34*u**4 + p33*u**3 + p32*u**2 + p31*u + p30)
    elif u > 46.0:
        znotm = 10 * (1/np.exp(np.sqrt(0.16 / (p45*u**5 + p44*u**4 + p43*u**3 + p42*u**2 + p41*u + p40))))
    else:
        raise ValueError(f'Wrong input uref value: {u}')
    
    roh = 1.2  # 공기 밀도, kg/m^3
    z0 = znotm
    
    U = np.sqrt(u**2)  # Wind speed
  
    Cd = (2.5 * np.log(10/z0))**-2
        
    Tx = Cd * U * roh * u
    
    print(Tx)
    return Tx

def sw_dens(S, T):
    T68 = T * 1.00024

    b0 =  8.24493e-1
    b1 = -4.0899e-3
    b2 =  7.6438e-5
    b3 = -8.2467e-7
    b4 =  5.3875e-9

    c0 = -5.72466e-3
    c1 = +1.0227e-4
    c2 = -1.6546e-6

    d0 = 4.8314e-4
    
    smow_T = sw.smow(T)
    dens = smow_T + (b0 + (b1 + (b2 + (b3 + b4 * T68) * T68) * T68) * T68) * S + (c0 + (c1 + c2 * T68) * T68) * S * np.sqrt(S) + d0 * (S ** 2)
    
    # print(f'smow_T : {smow_T}')
    
    return dens

def mixing_depth(D, T, S, Vmax, TS, R):
    # print(f'Depth : {D} \n Potential Temperature : {T} \n Salinity : {S} \n haitang wind speed : {Vmax} \n Translation speed : {TS} \n Radius : {R}')
    """
    Depth : D
    Potential Temperature : T
    Salinity : S
    Haitang wind speed : Vmax
    Translation speed : TS
    Radius : R 
    
    """
    Tx, Ty = ra_windstr(Vmax, 0) 
    # Tx = new_ra_windstr(Vmax) 
    # caculate wind stress, 해양 표층에 작용하는 힘을 나타내며, 수직 혼합에 영향을 준다 
    max_depth = np.floor(np.max(D))
    min_depth = np.min(D)
    depth_range = np.arange(min_depth, max_depth + 1, 1).astype(int)
    
    FT = 4*(R/TS) # residence time(s)
    p0 = 1024 # sea water density (kg/m3)
    SN = 1.2 # S number - nondimensional storm speed
    
    interp_T = interp1d(D, T, kind="linear", fill_value="extrapolate")
    interp_S = interp1d(D, S, kind="linear", fill_value="extrapolate")
    
    T1 = interp_T(depth_range)
    S1 = interp_S(depth_range)
        
    T1[:3] = T1[4]
    S1[:3] = S1[4]
    
    Dmix = 0
    Tmix = 0

    dens = sw_dens(S1, T1)
    
    for d in range(0, 5903):
        if (((9.8 * d) * (dens[d] - np.nanmean([dens[i] for i in range(0, d)]))) / (p0 * (SN * (Tx / (p0 * d)) * FT) ** 2)) >= 0.6:
            Dmix = d
            # 결과를 시각화
            break    
    Tmix = np.nanmean(T1[:Dmix])  
    
    return Dmix, Tmix, dens, FT, Tx

def mixing_depth_numpy_interp(D, T, S, Vmax, TS, R):
    # print(f'Depth : {D} \n Potential Temperature : {T} \n Salinity : {S} \n haitang wind speed : {Vmax} \n Translation speed : {TS} \n Radius : {R}')
   
    Tx, Ty = ra_windstr(Vmax, 0) 
    # caculate wind stress, 해양 표층에 작용하는 힘을 나타내며, 수직 혼합에 영향을 준다 
    
    max_depth = np.floor(np.max(D))
    min_depth = np.min(D)
    depth_range = np.arange(min_depth, max_depth + 1, 1).astype(int)
    
    FT = 4*(R/TS) # residence time(s)
    p0 = 1024 # sea water density (kg/m3)
    SN = 1.2 # S number - nondimensional storm speed
    
    T1 = np.interp(depth_range, D, T)
    S1 = np.interp(depth_range, D, S)
    
    T1[:3] = T1[4]
    S1[:3] = S1[4]
    
    Dmix = 0
    Tmix = 0

    dens = sw_dens(S1, T1)
    
    for d in range(0, 5903):
        
        if (((9.8 * d) * (dens[d] - np.nanmean([dens[i] for i in range(0, d)]))) / (p0 * (SN * (Tx / (p0 * d)) * FT) ** 2)) >= 0.6:
            Dmix = d
            break    
    Tmix = np.nanmean(T1[:Dmix])  
    
    return Dmix, Tmix, FT, Tx, dens, S1

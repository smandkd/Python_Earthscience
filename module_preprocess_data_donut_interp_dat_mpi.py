import numpy as np
from scipy.interpolate import interp1d
import seawater as sw
import os
import xarray as xr
from scipy.spatial import cKDTree
import numpy as np
import numba as nb
from MPI_ex.python_sang_ex.sang_ex.pyPI import constants
from MPI_ex.python_sang_ex.sang_ex.pyPI import utilities

base_salt_dict = r'/home/data/ECCO2/3DAYMEAN/SALT/'
fileEx = r'.nc'
base_opt_dict = r'/home/data/ECCO2/3DAYMEAN/THETA/'
dis_thres = 200/111

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

def points_200(lat, lon, tc_coords, index):
    coords = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)
    tree = cKDTree(coords)
    tc_coord = tc_coords[index]
    
    indices = tree.query_ball_point(tc_coord, dis_thres)
    
    points_200 = coords[indices]
    
    new_lat = points_200[:, 0]
    new_lon = points_200[:, -1]
    
    return new_lat, new_lon

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
    
def open_dataset(date):
    salt_dict = find_matching_salt_file(date, base_salt_dict)
    opt_dict = find_matching_opt_file(date, base_opt_dict)
    
    dataset_1 = xr.open_dataset(salt_dict)
    dataset_2 = xr.open_dataset(opt_dict)
    
    dataset_1['LONGITUDE_T'] = dataset_1['LONGITUDE_T'] - 180
    dataset_2['LONGITUDE_T'] = dataset_2['LONGITUDE_T'] - 180
    
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
    
    return donut_lon, donut_lat

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

def isel_time_dataset(data, index):
    result = data.isel(time=index, drop=True)
    return result

def isel_level_dataset(data, index):
    result = data.isel(level=index, drop=True)
    return result

def sel_lat_lon_dataset(data, lat, lon):
    result = data.sel(latitude=lat, longitude=lon, drop=True)
    return result

@nb.njit()
def cape(TP,RP,PP,T,R,P,ascent_flag=0,ptop=50,miss_handle=1):
#           P,TC,R: One-dimensional arrays  
#             containing pressure (hPa), 
#             temperature (C),
#             mixing ratio (g/kg).
    valid_i=~np.isnan(T)
    first_valid=np.where(valid_i)[0][0]

    # Are there missing values? If so, assess according to flag
    if (np.sum(valid_i) != len(P)):

        if (miss_handle != 0):
            CAPED=np.nan
            TOB=np.nan
            LNB=np.nan
            IFLAG=3
            print('IFLAGS 3\n')
            return(CAPED,TOB,LNB,IFLAG)

        else:
            if np.sum(np.isnan(T[first_valid:len(P)])>0):
                CAPED=np.nan
                TOB=np.nan
                LNB=np.nan
                IFLAG=3
                print('IFLAGS 3\n')
                return(CAPED,TOB,LNB,IFLAG)

            else:
                print('first_lvl=first_valid')
                first_lvl=first_valid
    else:
        print('first_lvl=0')
        first_lvl=0

    N=np.argmin(np.abs(P-ptop))
    
    P=P[first_lvl:N]
    T=T[first_lvl:N]
    R=R[first_lvl:N]
    nlvl=len(P)
    TVRDIF = np.zeros((nlvl,))
    
    if ((RP < 1e-6) or (TP < 200)):
        CAPED=0
        TOB=np.nan
        LNB=np.nan
        IFLAG=0
        print(f'RP : {RP}\n TP : {TP}\n')
        return(CAPED,TOB,LNB,IFLAG)

    TPC=utilities.T_ktoC(TP)                 # Parcel temperature in Celsius
    ESP=utilities.es_cc(TPC)                # Parcel's saturated vapor pressure
    EVP=utilities.ev(RP,PP)                 # Parcel's partial vapor pressure
    RH=EVP/ESP                              # Parcel's relative humidity
    S=utilities.entropy_S(TP,RP,PP)
    
    print(f'TPC : {TPC}\n ESP : {ESP}\n EVP : {EVP}\n')
    
    PLCL=utilities.e_pLCL(TP,RH,PP)
    
    CAPED=0
    TOB=T[0]
    IFLAG=1

    NCMAX=0
    jmin=int(1e6)
    
    for j in range(nlvl):
        
        jmin=int(min([jmin,j]))
    
        if (P[j] >= PLCL):
            TG=TP*(P[j]/PP)**(constants.RD/constants.CPD)
            RG=RP
            TLVR=utilities.Trho(TG,RG,RG)
            TVENV=utilities.Trho(T[j],R[j],R[j])
            TVRDIF[j,]=TLVR-TVENV
            
        else:
            
            TGNEW=T[j]
            TJC=utilities.T_ktoC(T[j])
            ES=utilities.es_cc(TJC)
            RG=utilities.rv(ES,P[j])
            
            NC=0
            TG=0

            while ((np.abs(TGNEW-TG)) > 0.001):
            
                TG=TGNEW
                TC=utilities.T_ktoC(TG)
                ENEW=utilities.es_cc(TC)
                RG=utilities.rv(ENEW,P[j])
                
                NC += 1
                
                ALV=utilities.Lv(TC)
                SL=(constants.CPD+RP*constants.CL+ALV*ALV*RG/(constants.RV*TG*TG))/TG
                EM=utilities.ev(RG,P[j])
                SG=(constants.CPD+RP*constants.CL)*np.log(TG)-constants.RD*np.log(P[j]-EM)+ALV*RG/TG
                if (NC < 3):
                    AP=0.3
                else:
                    AP=1.0
                TGNEW=TG+AP*(S-SG)/SL
                
                if (NC > 500) or (ENEW > (P[j]-1)):
                    CAPED=0
                    TOB=T[0]
                    LNB=P[0]
                    IFLAG=2
                    return(CAPED,TOB,LNB,IFLAG)
                
                NCMAX=NC
                
            RMEAN=ascent_flag*RG+(1-ascent_flag)*RP
            TLVR=utilities.Trho(TG,RMEAN,RG)
            TENV=utilities.Trho(T[j],R[j],R[j])
            TVRDIF[j,]=TLVR-TENV

    NA=0.0
    PA=0.0
    
    INB=0
    for j in range(nlvl-1, jmin, -1):
        if (TVRDIF[j] > 0):
            INB=max([INB,j])
            
    if (INB==0):
        CAPED=0
        TOB=T[0]
        LNB=P[INB]
        LNB=0
        return(CAPED,TOB,LNB,IFLAG)
    
    else:
    
        for j in range(jmin+1, INB+1, 1):
            PFAC=constants.RD*(TVRDIF[j]+TVRDIF[j-1])*(P[j-1]-P[j])/(P[j]+P[j-1])
            PA=PA+max([PFAC,0.0])
            NA=NA-min([PFAC,0.0])

        PMA=(PP+P[jmin])
        PFAC=constants.RD*(PP-P[jmin])/PMA
        PA=PA+PFAC*max([TVRDIF[jmin],0.0])
        NA=NA-PFAC*min([TVRDIF[jmin],0.0])
        
        PAT=0.0
        TOB=T[INB]
        LNB=P[INB]
        if (INB < nlvl-1):
            PINB=(P[INB+1]*TVRDIF[INB]-P[INB]*TVRDIF[INB+1])/(TVRDIF[INB]-TVRDIF[INB+1])
            LNB=PINB
            PAT=constants.RD*TVRDIF[INB]*(P[INB]-PINB)/(P[INB]+PINB)
            TOB=(T[INB]*(PINB-P[INB+1])+T[INB+1]*(P[INB]-PINB))/(P[INB]-P[INB+1])
    
        CAPED=PA+PAT-NA
        CAPED=max([CAPED,0.0])
        IFLAG=1

        return(CAPED,TOB,LNB,IFLAG)


@nb.njit()
def pi(SSTC,MSL,P,TC,R,CKCD=0.9,ascent_flag=0,diss_flag=1,V_reduc=0.8,ptop=50,miss_handle=1):
    

    # convert units
    SSTK=utilities.T_Ctok(SSTC) # SST in kelvin
    T=utilities.T_Ctok(TC)      # Temperature profile in kelvin
    R=R*0.001                   # Mixing ratio profile in g/g

    # CHECK 1: do SSTs exceed 5C? If not, set IFL=0 and return missing PI
    if (SSTC <= 5.0):
        VMAX=np.nan
        PMIN=np.nan
        IFL=0
        TO=np.nan
        OTL=np.nan
        return(VMAX,PMIN,IFL,TO,OTL)

    # CHECK 2: do Temperature profiles exceed 100K? If not, set IFL=0 and return missing PI
    if (np.min(T) <= 100):
        VMAX=np.nan
        PMIN=np.nan
        IFL=0
        TO=np.nan
        OTL=np.nan
        return(VMAX,PMIN,IFL,TO,OTL)
    
    # Set Missing mixing ratios to zero g/g, following Kerry's BE02 algorithm
    R[np.isnan(R)]=0.
    
    # Saturated water vapor pressure
    # from Clausius-Clapeyron relation/August-Roche-Magnus formula
    ES0=utilities.es_cc(SSTC)

    # define the level from which parcels lifted (first pressure level)
    NK=0
    
    #
    #   ***   Find environmental CAPE *** 
    #
    TP=T[NK]
    RP=R[NK]
    PP=P[NK]
    result = cape(TP,RP,PP,T,R,P,ascent_flag,ptop,miss_handle)
    CAPEA = result[0]
    IFLAG = result[3]
    # if the CAPE function tripped a flag, set the output IFL to it
    if (IFLAG != 1):
        IFL=int(IFLAG)
    
    #
    #   ***   Begin iteration to find mimimum pressure   ***
    #
    
    # set loop counter and initial condition
    NP=0         # loop counter
    PM=970.0
    PMOLD=PM     # initial condition from minimum pressure
    PNEW=0.0     # initial condition from minimum pressure
    IFL=int(1)   # Default flag for CAPE calculation

    # loop until convergence or bail out
    while (np.abs(PNEW-PMOLD) > 0.5):
        
        #
        #   ***  Find CAPE at radius of maximum winds   ***
        #
        TP=T[NK]
        PP=min([PM,1000.0])
        # find the mixing ratio with the average of the lowest level pressure and MSL
        RP=constants.EPS*R[NK]*MSL/(PP*(constants.EPS+R[NK])-R[NK]*MSL)
        result = cape(TP,RP,PP,T,R,P,ascent_flag,ptop,miss_handle)
        CAPEM = result[0]
        IFLAG = result[3]
        # if the CAPE function tripped a different flag, set the output IFL to it
        if (IFLAG != 1):
            IFL=int(IFLAG)
        
        #
        #  ***  Find saturation CAPE at radius of maximum winds    ***
        #  *** Note that TO and OTL are found with this assumption ***
        #
        TP=SSTK
        PP=min([PM,1000.0])
        RP=utilities.rv(ES0,PP)
        result = cape(TP,RP,PP,T,R,P,ascent_flag,ptop,miss_handle)
        CAPEMS, TOMS, LNBS, IFLAG = result
        # if the CAPE function tripped a flag, set the output IFL to it
        if (IFLAG != 1):
            IFL=int(IFLAG)
        # Store the outflow temperature and level of neutral bouyancy at the outflow level (OTL)
        TO=TOMS   
        OTL=LNBS
        # Calculate the proxy for TC efficiency (BE02, EQN. 1-3)
        RAT=SSTK/TO
        # If dissipative heating is "off", TC efficiency proxy is set to 1.0 (BE02, pg. 3)
        if (diss_flag == 0):
            RAT=1.0
        
        #
        #  ***  Initial estimate of pressure at the radius of maximum winds  ***
        #
        RS0=RP
        # Lowest level and Sea-surface Density Temperature (E94, EQN. 4.3.1 and 6.3.7)
        TV0=utilities.Trho(T[NK],R[NK],R[NK])
        TVSST=utilities.Trho(SSTK,RS0,RS0)
        # Average Surface Density Temperature, e.g. 1/2*[Tv(Tsfc)+Tv(sst)]
        TVAV=0.5*(TV0+TVSST)
        # Converge toward CAPE*-CAPEM (BE02, EQN 3-4)
        CAT=(CAPEM-CAPEA)+0.5*CKCD*RAT*(CAPEMS-CAPEM)
        CAT=max([CAT,0.0])
        # Iterate on pressure
        PNEW=MSL*np.exp(-CAT/(constants.RD*TVAV))
        
        #
        #   ***  Test for convergence (setup for possible next while iteration)  ***
        #
        # store the previous step's pressure       
        PMOLD=PM
        # store the current step's pressure
        PM=PNEW
        # increase iteration count in the loop
        NP += 1
        
        #
        #   ***   If the routine does not converge, set IFL=0 and return missing PI   ***
        #
        if (NP > 200)  or (PM < 400):
            VMAX=np.nan
            PMIN=np.nan
            IFL=0
            TO=np.nan
            OTL=np.nan
            return(VMAX,PMIN,IFL,TO,OTL)
    
    # Once converged, set potential intensity at the radius of maximum winds
    CATFAC=0.5*(1.+1/constants.b)
    CAT=(CAPEM-CAPEA)+CKCD*RAT*CATFAC*(CAPEMS-CAPEM)
    CAT=max([CAT,0.0])
    
    # Calculate the minimum pressure at the eye of the storm
    # BE02 EQN. 4
    PMIN=MSL*np.exp(-CAT/(constants.RD*TVAV))
                 
    # Calculate the potential intensity at the radius of maximum winds
    # BE02 EQN. 3, reduced by some fraction (default 20%) to account for the reduction 
    # of 10-m winds from gradient wind speeds (Emanuel 2000, Powell 1980)
    FAC=max([0.0,(CAPEMS-CAPEM)])
    VMAX=V_reduc*np.sqrt(CKCD*RAT*FAC) 
        
    # Return the calculated outputs to the above program level
    return(VMAX,PMIN,IFL,TO,OTL)

def run_sample_dataset(fn, CKCD):
    """ This function calculates PI over the sample dataset using xarray """
    
    ds = xr.open_dataset(fn)
    
    result = xr.apply_ufunc(
        pi,
        ds['sst'], ds['msl'], ds['level'].data, ds['t'], ds['q'],
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=50, miss_handle=1),
        input_core_dims=[
            [], [], ['level', ], ['level', ], ['level', ],
        ],
        output_core_dims=[
            [], [], [], [], []
        ],
        vectorize=True
    ) 
    
    # store the result in an xarray data structure
    vmax, pmin, ifl, t0, otl = result    
    out_ds=xr.Dataset({
        'vmax': vmax , 
        'pmin': pmin,
        'ifl': ifl,
        't0': t0,
        'otl': otl,
        # merge the state data into the same data structure
        'sst': ds.sst,
        't': ds.t,
        'q': ds.q,
        'msl': ds.msl,
        'lsm': ds.lsm,
        })
        
    return out_ds
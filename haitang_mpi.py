#%%
import xarray as xr
import pickle
from pyPI import pi
from pyPI.utilities import *
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
from pyPI import constants
from pyPI import utilities
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

def isel_time_dataset(data, index):
    result = data.isel(time=index, drop=True)
    return result

def isel_level_dataset(data, index):
    result = data.isel(level=index, drop=True)
    return result

def sel_lat_lon_dataset(data, lat, lon):
    result = data.sel(latitude=lat, longitude=lon, drop=True)
    return result
#%% Extracting Haitang from IBTrACS
dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp'), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True) # Haitang 2005

wind_array = tc_HAITANG.usa_wind.data[0]
gen_index = gen(wind_array)
lmi_index = lmi(wind_array)
selected_indices = np.arange(gen_index, lmi_index+1)
haitang_dataset = tc_HAITANG.isel(date_time=selected_indices, drop=True)

days_3days_before_revised = haitang_dataset['time'] - pd.Timedelta(days=3)
haitang_time = days_3days_before_revised['time']
haitang_time
# Substraction 3 days in  variable 'time' 
dt_array = []
for data in haitang_time[0]:
    datetime = np.array(data, dtype='datetime64[ns]')
    time = np.datetime64(datetime, 'D')
    dt_array.append(time)

lat = np.array(haitang_dataset.usa_lat[0], dtype=object)
lon = np.array(haitang_dataset.usa_lon[0], dtype=object)
# Haitang usa latitude, longitude 
HAITANG_coords = np.column_stack((lat,lon))
HAITANG_coords
#%% Air Temp, Specific Hum, MSLP, SST from ERA5, OISST 
airt_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/airt/era5_pres_temp_200507.nc')
shum_dataset = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/shum/era5_pres_spec_200507.nc')
dataset_era5_mslp = xr.open_dataset('/home/data/ECMWF/ERA5/WP/6hourly/mslp/era5_surf_mslp_200507.nc')
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')

#%% level array
level_arr = airt_dataset.level.data[::-1] # sorting downgrade 

#%% dis threshold, fitering dataset with dt_array

dis_thres = 200/111

before_3_shum_dataset = shum_dataset.sel(time=dt_array, drop=True)
days_3_before_oisst = oisst_DATA.sel(time=dt_array, drop=True)
days_3_before_mslp = dataset_era5_mslp.sel(time=dt_array, drop=True)
days_3_before_airt = airt_dataset.sel(time=dt_array, drop=True)

#%%
# -------------------
#
# Specific Humidity
#
# -------------------
shum_mean_arr = []

for index, time in enumerate(dt_array):
    shum_arr = []
    
    shum_dataset = isel_time_dataset(before_3_shum_dataset, index)
    
    lat = shum_dataset.latitude
    lon = shum_dataset.longitude
    shum_coords = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)
    shum_tree = cKDTree(shum_coords)
    haitang_coord = HAITANG_coords[index]
    
    indices = shum_tree.query_ball_point(haitang_coord, dis_thres)
    
    shum_200_points = shum_coords[indices]
    
    new_lat = shum_200_points[:, 0]
    new_lon = shum_200_points[:, -1]
    len_level = len(shum_dataset.level.data)
    filtered_shum_dataset = sel_lat_lon_dataset(shum_dataset, new_lat, new_lon)
    
    for level in range(len_level):
        dataset = isel_level_dataset(filtered_shum_dataset, level)
        
        shum = dataset.q.data
        shum_mean = np.mean(shum)
        shum_arr.append(shum_mean)
        
        
    shum_mean_arr.append(shum_arr[::-1])
#%%
# -------------------------
#
# Sea Sureface Temperature
#
# -------------------------
sst_daily_averages = []

for index, time in enumerate(dt_array):
    oisst_dataset = isel_time_dataset(days_3_before_oisst, index)

    oisst_lat = oisst_dataset.lat
    oisst_lon = oisst_dataset.lon - 180
    oisst_sst = oisst_dataset.sst.data.flatten()
    oisst_coords = np.array(np.meshgrid(oisst_lat, oisst_lon)).T.reshape(-1, 2)
    oisst_tree = cKDTree(oisst_coords)
    oisst_coords_sst = np.column_stack((oisst_coords, oisst_sst))
    haitang_index_coord = HAITANG_coords[index]
    
    indices = oisst_tree.query_ball_point(haitang_index_coord, dis_thres)
    oisst_200_points = oisst_coords_sst[indices]
    sst_average = np.mean(oisst_200_points[:, -1])  # 마지막 열(SST)에 대한 평균
    sst_daily_averages.append(sst_average)

#%%
# -----------------------
# Mean sea level pressure
# Pa to hPa (1hPa = 100Pa, divide 100)
# -----------------------
mslp_daily_averages = []

for index, time in enumerate(dt_array):
    mslp_dataset = isel_time_dataset(days_3_before_mslp, index)
    
    mslp_lat = mslp_dataset.latitude
    mslp_lon = mslp_dataset.longitude
    mslp_value = (mslp_dataset.msl.data/100).flatten()
    mslp_coords = np.array(np.meshgrid(mslp_lat, mslp_lon)).T.reshape(-1, 2)
    mslp_tree = cKDTree(mslp_coords)
    coords_msl = np.column_stack((mslp_coords, mslp_value))
    haitang_index_coord = HAITANG_coords[index]

    indices = mslp_tree.query_ball_point(haitang_index_coord, dis_thres)
    msl_200_points = coords_msl[indices]
    msl = msl_200_points[:, -1]
    msl_ave = np.mean(msl)
    mslp_daily_averages.append(msl_ave)

#%%
# -----------------------
#
# Air Temperature 
#
# t's unit is K, so change to degC ( - 273.15 )
#
# -----------------------
array = []

for index, time in enumerate(dt_array):
    level_airt_arr = []
    
    airt_dataset = isel_time_dataset(days_3_before_airt, index)
    lat = airt_dataset.latitude
    lon = airt_dataset.longitude
    airt_coords = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)
    airt_tree = cKDTree(airt_coords)
    haitang_coord = HAITANG_coords[index]
    
    indices = airt_tree.query_ball_point(haitang_coord, dis_thres)
    
    airt_200_points = airt_coords[indices]
    
    lat = airt_200_points[:,0]
    lon = airt_200_points[:,-1]
    new_airt_data = sel_lat_lon_dataset(airt_dataset, lat, lon)
    
    for level in range(len(airt_dataset.level.data)):
        dataset = isel_level_dataset(new_airt_data, level)
        airt_arr = dataset.t.data - 273.15
        airt_mean = np.mean(airt_arr)
        level_airt_arr.append(airt_mean)
        
    array.append(level_airt_arr[::-1])

# %%
dims = dict(
    time = dt_array,
    level = level_arr
)

lsm_arr = np.ones((len(dt_array), len(level_arr)))

data_vars = {
    'lsm': (['time', 'level'], lsm_arr),
    't': (['time', 'level'], array),
    'q': (['time', 'level'], shum_mean_arr),
    'msl': (['time'], mslp_daily_averages),
    'sst' : (['time'], sst_daily_averages),
}

#%%
dataset = xr.Dataset(data_vars, coords=dims)
nc_path = '0307_data.nc'
dataset.to_netcdf(path=nc_path)
dt = xr.open_dataset('0307_data.nc')
# %% pi method
# define the function to calculate CAPE
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

#%% PI

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

#%%

def run_sample_dataset(fn, dim='p',CKCD=0.9):
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
# %%
df = '0307_data.nc'
ds = run_sample_dataset(df)
ds.to_netcdf('0307_output.nc')

# %%
output_dataset = xr.open_dataset('0307_output.nc')
mpi = output_dataset.vmax.data
# %%
mpi * 1.94384 # m/s to knots 
"""
[49.27785991,  27.3536465 ,  29.9748774 ,  35.49059987,
43.70909594,  28.90875975,  36.50847693,  51.33722912,
61.29692311,  56.55128267,  76.67114932,  88.74722826,
95.73483874,  82.26174305, 109.95686978, 126.97691888,
139.46652577, 131.02596132, 145.79907945, 154.53116968]

[ 35.,  35.,  35.,  35.,
35.,  35.,  50.,  65.,
70.,  75.,  75.,  85., 
90.,  90., 100., 115., 
120., 130., 135., 140.]

"""
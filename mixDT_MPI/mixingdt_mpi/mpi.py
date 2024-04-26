import numpy as np
import numba as nb
from pyPI import constants
from pyPI import utilities
import xarray as xr

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
                first_lvl=first_valid
    else:
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
        
        return(CAPED,TOB,LNB,IFLAG)

    TPC=utilities.T_ktoC(TP)                 # Parcel temperature in Celsius
    ESP=utilities.es_cc(TPC)                # Parcel's saturated vapor pressure
    EVP=utilities.ev(RP,PP)                 # Parcel's partial vapor pressure
    RH=EVP/ESP                              # Parcel's relative humidity
    S=utilities.entropy_S(TP,RP,PP) 
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
    
    if (SSTC <= 5.0):
        VMAX=np.nan
        PMIN=np.nan
        IFL=0
        TO=np.nan
        OTL=np.nan
        return(VMAX,PMIN,IFL,TO,OTL)

    if (np.min(T) <= 100):
        VMAX=np.nan
        PMIN=np.nan
        IFL=0
        TO=np.nan
        OTL=np.nan
        return(VMAX,PMIN,IFL,TO,OTL)
    
    R[np.isnan(R)]=0.
    
    ES0=utilities.es_cc(SSTC)

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
        RP=constants.EPS*R[NK]*MSL/(PP*(constants.EPS+R[NK])-R[NK]*MSL)
        result = cape(TP,RP,PP,T,R,P,ascent_flag,ptop,miss_handle)
        
        CAPEM = result[0]
        IFLAG = result[3]
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
        print('cape ')
        print(result)
        CAPEMS, TOMS, LNBS, IFLAG = result
        
        if (IFLAG != 1):
            IFL=int(IFLAG)
        TO=TOMS   
        OTL=LNBS
        # Calculate the proxy for TC efficiency (BE02, EQN. 1-3)
        RAT=SSTK/TO
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
        
        PMOLD=PM
        PM=PNEW
        NP += 1
        
        if (NP > 200)  or (PM < 400):
            VMAX=np.nan
            PMIN=np.nan
            IFL=0
            TO=np.nan
            OTL=np.nan
            return(VMAX,PMIN,IFL,TO,OTL)
    
    CATFAC=0.5*(1.+1/constants.b)
    CAT=(CAPEM-CAPEA)+CKCD*RAT*CATFAC*(CAPEMS-CAPEM)
    CAT=max([CAT,0.0])
    
    PMIN=MSL*np.exp(-CAT/(constants.RD*TVAV))
                 
    # Calculate the potential intensity at the radius of maximum winds
    FAC=max([0.0,(CAPEMS-CAPEM)])
    VMAX=V_reduc*np.sqrt(CKCD*RAT*FAC) 
        
    # Return the calculated outputs to the above program level
    return(VMAX,PMIN,IFL,TO,OTL)

def run_sample_dataset(fn, CKCD):
    """ This function calculates PI over the sample dataset using xarray """
    
    ds = xr.open_dataset(fn)
    
    result = xr.apply_ufunc(
        pi,
        ds['sst'], ds['mslp'], ds['level'].data, ds['airt'], ds['shum'],
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=50, miss_handle=1),
        input_core_dims=[
            [], [], ['level', ], ['level', ], ['level', ],
        ],
        output_core_dims=[
            [], [], [], [], []
        ],
        vectorize=True
    ) 
    
    vmax, pmin, ifl, t0, otl = result    
    out_ds=xr.Dataset({
        'vmax': vmax , 
        'pmin': pmin,
        'ifl': ifl,
        't0': t0,
        'otl': otl,
        'sst': ds.sst,
        'airt': ds.airt,
        'shum': ds.shum,
        'mslp': ds.mslp,
        'lsm': ds.lsm,
        })
        
    return out_ds

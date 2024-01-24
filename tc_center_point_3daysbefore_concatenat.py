# HAITANG Genesis : 2005-07-11 18:00:00 , sst : 2005-07-09 
#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cartopy.crs as crs
import pandas as pd
#%%
dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
# drop = True 로 할 경우 결측값을 결과 배열에서 완전히 제거한다. 
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp') & (filtered_DATA.usa_wind >= 34), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True)
GEN_index = np.where(tc_HAITANG.usa_wind >= 34)[1][0]
print(GEN_index)
LMI_index = np.where(tc_HAITANG.usa_wind == tc_HAITANG.usa_wind.max())[1][-1]
HAITANG_gen2lmi_lat = tc_HAITANG.usa_lat[0][GEN_index:LMI_index+1]
HAITANG_gen2lmi_lon = tc_HAITANG.usa_lon[0][GEN_index:LMI_index+1]
HAITANG_coord = np.array(list(zip(HAITANG_gen2lmi_lat, HAITANG_gen2lmi_lon)))
HAITANG_gen = HAITANG_coord[0]
# %%

HAITANG_gen2lmi = tc_HAITANG.isel(date_time=slice(GEN_index, LMI_index+1), drop=True)
HAITANG_gen2lmi
hr_00_data = HAITANG_gen2lmi.where(HAITANG_gen2lmi.time.dt.hour == 00, drop=True)
lat = np.array(hr_00_data.usa_lat[0], dtype=object)
lon = np.array(hr_00_data.usa_lon[0], dtype=object)
print(lat)

# %%

days_3_before = hr_00_data['time'] - pd.Timedelta(days=3)
time_array = np.array(days_3_before[0])
print(time_array)

# %%

HAITANG_00_coords_with_days3before = np.column_stack((lat,lon,time_array))
HAITANG_00_coords_with_days3before


#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cartopy.crs as crs 
import cartopy.feature as cfeature

# %%

dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)

    
# %%
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp') & (filtered_DATA.usa_wind >= 34), drop=True)
jtwc_DATA
# %%
time_array = jtwc_DATA.time
sid_array = jtwc_DATA.sid
lat_array = jtwc_DATA.usa_lat
lon_array = jtwc_DATA.usa_lon
usa_speed_array = jtwc_DATA.usa_wind
print(time_array)
print(sid_array)
# %%

tc_1 = jtwc_DATA.isel(storm=100)
tc_haitang = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True)
# %%
tc_haitang.usa_wind
lmi_INDEX = np.where(tc_haitang.usa_wind == tc_haitang.usa_wind.max())[-1]
lmi_INDEX



# %%
tc_1.sid # 2004074N03143
tc_1.name
# %%
print(tc_1.usa_wind)
print(len(tc_1.usa_wind)) # 137
"""
[20., nan, 20., nan, 20., nan, 20., nan, 20., nan, 20., nan, 25.,
nan, 25., nan, 25., nan, 25., nan, 30., nan, 30., nan, 30., nan,
30., nan, 30., nan, 25., nan, 25., nan, 25., nan, 30., nan, 35.,
nan, 35., nan, 35., nan, 35., nan, 35., nan, 35., nan, 35., nan,
35., nan, 35., nan, 35., nan, 35., nan, 35., nan, 30., nan, 25.,
nan, 25., nan, 25., nan, 25., nan, 20., nan, 20., nan, 20., nan,
nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
"""
# %%
indexes = np.where(tc_1.usa_wind >= 34)
gen_INDEX = indexes[0][0] # 34
print(gen_INDEX)
# %%
lmi_INDEX = np.where(tc_1.usa_wind == tc_1.usa_wind.max())[0][-1]
lmi_INDEX # 56
# %%

len(tc_1.usa_wind[gen_INDEX:lmi_INDEX+1]) #23
print(tc_1.time[gen_INDEX:lmi_INDEX+1])
# %%
tc_1.lat
# %%

len(tc_1.lat[gen_INDEX:lmi_INDEX+1]) # 23

# %%
tc_1_usa_wind_GEN2LMI = tc_1.usa_wind[gen_INDEX:lmi_INDEX+1]
tc_1_lon_GEN2LMI = tc_1.lon[gen_INDEX:lmi_INDEX+1]
tc_1_lat_GEN2LMI = tc_1.lat[gen_INDEX:lmi_INDEX+1]
tc_1_time_GEN2LMI = tc_1.time[gen_INDEX:lmi_INDEX+1]
# %%

for case_index in range(2):
        tc_i = jtwc_DATA.isel(storm=case_index)
        gen_INDEX = np.where(tc_i.usa_wind >= 34)[0][0]
        lmi_INDEX = np.where(tc_i.usa_wind == tc_i.usa_wind.max())[0][-1]
        tc_i_usa_wind_GEN2LMI = tc_i.usa_wind[gen_INDEX:lmi_INDEX+1] 
        tc_i_lon_GEN2LMI = tc_i.lon[gen_INDEX:lmi_INDEX+1]
        tc_i_lat_GEN2LMI = tc_i.lat[gen_INDEX:lmi_INDEX+1]
        tc_i_time_GEN2LMI = tc_i.time[gen_INDEX:lmi_INDEX+1]
        # print(f'{gen_INDEX} \n{lmi_INDEX} \n{len(tc_i_usa_wind_GEN2LMI)} \n{len(tc_i_lon_GEN2LMI)} \n{len(tc_i_lat_GEN2LMI)} \n{len(tc_i_time_GEN2LMI)}')
        tc_i_coord = np.array(list(zip(tc_i_lat_GEN2LMI, tc_i_lon_GEN2LMI)))
        # print(f'tc sid : {tc_i.sid.data}, \ntc time : {tc_i_time_GEN2LMI.data}, \ntc usa_wind : {tc_i_usa_wind_GEN2LMI.data}')
        
        

# %%

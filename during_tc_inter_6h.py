#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cartopy.crs as crs 


# %%
dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) &
                              (dataset.season <= 2020), drop=True)
# drop = True 로 할 경우 결측값을 결과 배열에서 완전히 제거한다. 
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp') &
                                (filtered_DATA.usa_wind >= 34), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True)

# %%

GEN_index = np.where(tc_HAITANG.usa_wind >= 34)[1][0]
LMI_index = np.where(tc_HAITANG.usa_wind == tc_HAITANG.usa_wind.max())[1][-1]
HAITANG_gen2lmi_lat = tc_HAITANG.usa_lat[0][GEN_index:LMI_index+1]
HAITANG_gen2lmi_lon = tc_HAITANG.usa_lon[0][GEN_index:LMI_index+1]
HAITANG_coord = np.array(list(zip(HAITANG_gen2lmi_lat, HAITANG_gen2lmi_lon)))
HAITANG_gen2lmi = tc_HAITANG.isel(date_time=slice(GEN_index, LMI_index+1), drop=True)

# %%
interval_6_data = HAITANG_gen2lmi.where(HAITANG_gen2lmi.time[::4], drop=True)
days_3_before = interval_6_data['time']
days_3_before[0]

# %%

dt_array = []
for data in days_3_before[0]:
    datetime = np.array(data, dtype='datetime64[ns]')
    time = np.datetime64(datetime, 'D')
    dt_array.append(time)
    
dt_array

# %%
lat = np.array(interval_6_data.usa_lat[0], dtype=object)
lon = np.array(interval_6_data.usa_lon[0], dtype=object)
new_data_haitang = np.column_stack((lat, lon))
new_data_haitang
# %%
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')
days_3_before_oisst = oisst_DATA.sel(time=dt_array, drop=True)

# %%
plt.figure(figsize=(10,5))
ax = plt.axes(projection=crs.PlateCarree())
ax.coastlines(resolution='10m')
ax.set_extent([100, 155, 10, 48])
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

for index, data in enumerate(dt_array):
    sst_sum = 0
    oisst_07 = oisst_DATA.sel(time=data)
    oisst_07_lat = oisst_07.lat
    oisst_07_lon = oisst_07.lon - 180
    oisst_07_sst = oisst_07.sst.data.flatten()
    oisst_07_coords = np.array(np.meshgrid(oisst_07_lat, oisst_07_lon)).T.reshape(-1, 2)
    oisst_07_coords_tree = cKDTree(oisst_07_coords)
    oisst_07_coords_sst = np.column_stack((oisst_07_coords, oisst_07_sst))
    
    haitang_07_coord = new_data_haitang[index]
    dis_thres = 200/111
    
    indices = oisst_07_coords_tree.query_ball_point(haitang_07_coord, dis_thres)
    points_07 = oisst_07_coords_sst[indices]
    
    for j in range(len(points_07)):
        ax.plot(points_07[j][1], points_07[j][0], color="red", marker="o", markersize=0.5, transform=crs.PlateCarree())
        
plt.savefig('during_tc_inter6h.pdf')
plt.title('Points within a radius of 200km during tc intervals 6 hours')
plt.show()
    
# %%

#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as crs
from haversine import haversine 
from scipy.spatial import cKDTree 

# %%
dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')

filtered_data = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), 
                              drop=True)
jtwc_DATA = filtered_data.where((filtered_data.usa_agency == b'jtwc_wp'), drop=True)
tc_KALMAEGI = jtwc_DATA.where(jtwc_DATA.sid == b'2008193N20126', drop=True)

# %%

print(f'{tc_KALMAEGI.usa_lon}, {tc_KALMAEGI.usa_lat}, {tc_KALMAEGI.usa_wind}')

# %%

gen_INDEX = np.where(tc_KALMAEGI.usa_wind >= 34)[-1][0]
print(gen_INDEX)

# %%

lmi = tc_KALMAEGI.usa_wind.max()
lmi_INDEX = np.where(tc_KALMAEGI.usa_wind == lmi)[-1][1]
print(lmi_INDEX)

# %%

kalmaegi_GEN2LMI_lat = tc_KALMAEGI.usa_lat[0][gen_INDEX:lmi_INDEX+1]
kalmaegi_GEN2LMI_lon = tc_KALMAEGI.usa_lon[0][gen_INDEX:lmi_INDEX+1]
print(f'{kalmaegi_GEN2LMI_lat[0]}, {kalmaegi_GEN2LMI_lon[0]}')
# 2008-07-15 06:00:00

# %%

tc_KALMAEGI_coords = np.array(list(zip(kalmaegi_GEN2LMI_lat, kalmaegi_GEN2LMI_lon)))
tc_KALMAEGI_coords
# %%
oisst_DATA = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2008.nc')
oisst_DATA
# %%
data_20080715 = oisst_DATA.sel(time='2008-07-15').to_dataarray()
print(data_20080715.lat.values[0])
print(data_20080715.lon.values[0])

# %%

data_20080715_lat = data_20080715.lat.values
data_20080715_lon = data_20080715.lon.values - 180


data_20080715_coords = np.array(
    np.meshgrid(data_20080715_lat, data_20080715_lon)
    ).T.reshape(-1, 2)

data_20080715_coords

# %%

data_20080715_tree = cKDTree(data_20080715_coords)
# This class provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point
data_20080715_tree
# %%

distance_THRESHOLD = 200/111
# 거리 임계값 설정 : 200km 이내에 잇는 SST 데이터 포인트만 고려하도록 거리 임계값을 설정


# %%

indices = data_20080715_tree.query_ball_point(tc_KALMAEGI_coords[0], distance_THRESHOLD)

# %%
# tc_KALMAEGI_coords 가 하나일때 
nearby_SST_coords = data_20080715_coords[indices]

# %%

print(nearby_SST_coords)


#%%
plt.figure(figsize=(10,5))
ax = plt.axes(projection=crs.PlateCarree())
ax.coastlines(resolution='10m')
ax.set_extent([100, 180, -10, 40])
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# 점 표시
for i in range(len(nearby_SST_coords)):
    plt.scatter(x=nearby_SST_coords[i][1], y=nearby_SST_coords[i][0], 
         color="red", s=1, alpha=0.5, transform=crs.PlateCarree())
"""
# 해수면 온도 지도에 표시
sst_plot = ax.contourf(nearby_SST_coords[0:][0], nearby_SST_coords[0:][1], 
                       data_20080716_coords_WITH_sst[0:][2], 100,                     transforms=crs.PlateCarree(), cmap='coolwarm')
"""
plt.savefig('nearby_coords.pdf')
plt.show()
# %%

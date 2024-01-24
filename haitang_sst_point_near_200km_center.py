#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cartopy.crs as crs
# %%

dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
filtered_DATA = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
# drop = True 로 할 경우 결측값을 결과 배열에서 완전히 제거한다. 
jtwc_DATA = filtered_DATA.where((filtered_DATA.usa_agency == b'jtwc_wp') & (filtered_DATA.usa_wind >= 34), drop=True)
tc_HAITANG = jtwc_DATA.where(jtwc_DATA.sid == b'2005192N22155', drop=True)

# %%

GEN_index = np.where(tc_HAITANG.usa_wind >= 34)[1][0]
print(GEN_index)
LMI_index = np.where(tc_HAITANG.usa_wind == tc_HAITANG.usa_wind.max())[1][-1]
"""
첫 번째 배열은 첫 번째 차원(가령, 행)에서의 인덱스를 나타냅니다.
두 번째 배열은 두 번째 차원(가령, 열)에서의 인덱스를 나타냅니다.
"""

# %%

HAITANG_gen2lmi_lat = tc_HAITANG.usa_lat[0][GEN_index:LMI_index+1]
HAITANG_gen2lmi_lon = tc_HAITANG.usa_lon[0][GEN_index:LMI_index+1]
HAITANG_coord = np.array(list(zip(HAITANG_gen2lmi_lat, HAITANG_gen2lmi_lon)))
HAITANG_gen = HAITANG_coord[0]
# list : 다양한 데이터 타입을 python의 리스트 자료형으로 만든다.
# zip : 여러 개의 이터러블(iterable, ex : 리스트, 튜플)을 인자로 받아, 각 이터러블의 동일한 인덱스에 위치한 요소들을 튜플로 묶어서 반환해준다.

# %%

OISST_dataset = xr.open_dataset('/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.2005.nc')
OISST_dataset

# %%

oisst_data = OISST_dataset.sel(time = '2005-07-08', drop=True)
oisst_data
# %%
oisst_lat = oisst_data.lat
oisst_lon = oisst_data.lon - 180 
sst = oisst_data.sst.data.flatten()
oisst_coords = np.array( np.meshgrid( oisst_lat, oisst_lon ) ).T.reshape(-1, 2) 
oisst_coords_with_sst = np.column_stack( (oisst_coords ,sst) )
oisst_coords_with_sst
tree = cKDTree(oisst_coords)
tree
dis_THRE = 200/111
ind = tree.query_ball_point(HAITANG_gen, dis_THRE)
new_data = oisst_coords_with_sst[ind]
# %%

plt.figure(figsize=(10, 5))
ax = plt.axes(projection=crs.PlateCarree())
ax.coastlines(resolution='10m')
ax.set_extent([100, 155, 10, 40])
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

for i in range(len(new_data)):
    plt.scatter(x=new_data[i][1], y=new_data[i][0], color="red", s=1, alpha=0.5, transform=crs.PlateCarree())
    print(new_data[i][2])
    
plt.savefig('haitang_200_plot.pdf')
plt.show()


# %%

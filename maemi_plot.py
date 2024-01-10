#%%
import netCDF4 as nc

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature 

# 데이터 셋 열기
dataset = nc.Dataset('IBTrACS.WP.v04r00.nc')


name_data = dataset.variables['name'][:] # name 배열 
lat = dataset.variables['lat'] # latitude(위도) 배열
lon = dataset.variables['lon'] # longitude(경도) 배열 

# 이름을 저장할 리스트 초기화
names = []
current_name = ''

# 바이트 배열을 순회하면서 이름 추출
for byte_array in name_data:
    # 각 바이트 배열을 문자열로 변환 
    name_str = ''.join(byte.decode('utf-8') for byte in byte_array if byte != b'')
    
    # 변환된 문자열을 리스트에 추가 
    if name_str:
        names.append(name_str)
            
if 'MAEMI' in names:
    maemi_index = names.index('MAEMI')
    maemi_lat = dataset.variables['lat'][maemi_index][:]
    maemi_lon = dataset.variables['lon'][maemi_index][:]
    
    # print(lat[maemi_index][:])
    # print(lon[maemi_index][:])
else:
    print('MAEMI not found ind the dataset')       
    
print(maemi_lat)
print(maemi_lon)
 
# 새로운 그래프 창 생성    
fig = plt.figure(figsize=(10, 6))
# 그래프 창에 새로운 축(subplot) 추가. '1,1,1'은 1x1 격자에 첫번째 subplot을 의미하며, 'projection=crs.Robinson()'은 로빈슨 투영법을 사용하여 지도를 그릴 것임을 지정. 
ax = fig.add_subplot(1,1,1, projection=crs.Robinson())
# 지도의 범위를 설정한다. 
ax.set_extent([100, 180, 7, 60])
ax.add_feature(cfeature.COASTLINE, edgecolor="tomato") # 해안선 추가
ax.add_feature(cfeature.BORDERS, edgecolor="tomato") # 국경 추가 
ax.gridlines() # 지도에 격자선 추가
ax.add_feature(cfeature.LAND) # 땅 추가
ax.add_feature(cfeature.OCEAN) # 바다 추가 

"""
plt.scatter : 지도에 점들을 표시하는데 사용한다. 
color="black" : 점의 색상을 검정색으로 설정한다.
s=1 : 점의 크기를 1로 설정한다. 
alpha=0.5 : 점의 투명도를 50%로 설정한다. 
transform=crs.PlateCarree() : 점의 좌표계를 지정한다. 여기서 platcarree() 일반적인 지도의 좌표게이다. 
"""
plt.scatter(x=maemi_lon, y=maemi_lat, color="black", s=1, alpha=0.5, transform=crs.PlateCarree())
plt.savefig("maemi_path.pdf")
plt.show()

# %%

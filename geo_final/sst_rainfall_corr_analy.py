#%%
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import griddata
import scipy.stats as stats 
# %%
ds = xr.open_dataset('/home/tkdals/Python_Earthscience/geo_final_exam/sst.mnmean.v4.nc')

# %%
lons = ds.lon.data
lats = ds.lat.data
kor_lats = []
kor_lons = []
extent = [117, 138, 25, 45]

for lon in lons:
    if (lon >= 117) & (lon <= 138):
        kor_lons.append(lon)

for lat in lats:
    if (lat >= 25) & (lat <=45):
        kor_lats.append(lat)
 
kor_lats = kor_lats[::-1]

kor_ds = ds.sel(lon=kor_lons, lat=kor_lats, drop=True)
kor_sst = kor_ds.sst.data

# %%
# ================================================
#  지도에 sst 표시 
# ================================================
kor_2020 = kor_ds.sel(time=kor_ds.time.data[0], drop=True)
kor_sst_2020 = kor_2020.sst.data
# 1차원 배열로 변환
lon, lat = np.meshgrid(kor_lons, kor_lats)
lon = lon.flatten()
lat = lat.flatten()
sst = kor_sst_2020.flatten()

# 보간할 그리드 생성
grid_lon = np.linspace(extent[0], extent[1], 500)
grid_lat = np.linspace(extent[2], extent[3], 500)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 유효한 SST 값이 있는 지점 선택
valid_points = ~np.isnan(sst)
valid_lon = lon[valid_points]
valid_lat = lat[valid_points]
valid_sst = sst[valid_points]

# 보간 수행
grid_sst = griddata((valid_lon, valid_lat), valid_sst, (grid_lon, grid_lat), method='linear')

# 육지를 마스킹하기 위한 기능 추가
land_mask = cfeature.NaturalEarthFeature(
    'physical', 'land', '10m', edgecolor='black',
    facecolor='white')

# 지도 시각화
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([120, 133, 29, 38])

# 해양과 육지, 국가 경계선 추가
ax.add_feature(cfeature.COASTLINE, zorder=1, color='black')
ax.add_feature(cfeature.BORDERS, zorder=1)
ax.add_feature(land_mask, zorder=2)

# 보간된 SST 데이터 시각화 (육지를 마스킹)
masked_sst = np.ma.masked_where(np.isnan(grid_sst), grid_sst)
c = ax.pcolormesh(grid_lon, grid_lat, masked_sst, transform=ccrs.PlateCarree(), cmap='coolwarm', zorder=1)

# 색상 막대 추가
fig.colorbar(c, ax=ax, orientation='vertical', label='Sea Surface Temperature (°C)')

plt.title('2020 Interpolated Sea Surface Temperature around South Korea')
plt.show()
# %%
# ================================================
#   sst 그래프 
# ================================================
jeju_ds = ds.sel(lon = 128, lat = 32, drop=True)
# 시간 데이터 추출
time_data = jeju_ds.time.data

# 필터링 기준 설정
start_date = np.datetime64('1980-01-01')
end_date = np.datetime64('2020-12-31')

# 조건에 맞는 데이터 필터링
filtered_time_indices = (time_data >= start_date) & (time_data <= end_date)
filtered_time_data = time_data[filtered_time_indices]

# 데이터셋에서 필터링된 시간에 해당하는 데이터 선택
filtered_ds = jeju_ds.sel(time=filtered_time_data)

jeju_sst_1980_2020 = filtered_ds.sst

plt.figure(figsize=(12, 6))
plt.plot(filtered_time_data, jeju_sst_1980_2020, label='Sea Surface Temperature')

plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Sea Surface Temperature from 1980 to 2020')
plt.legend()
plt.grid(True)
plt.show()
# %%
# ================================================
#   강수량 그래프 
# ================================================
file_path = '/home/tkdals/Python_Earthscience/geo_final_exam/rn_20240607155457.csv'
data = pd.read_csv(file_path, encoding='utf-8')

rainfall_jeju = data.iloc[:, 2]
year_month_jeju = data.iloc[:, 0]

plt.figure(figsize=(12, 6))
plt.plot(filtered_time_data, rainfall_jeju, label='Rainfall jeju')

plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Sea Surface Temperature from 1980 to 2020')
plt.show()
# %%
X = jeju_sst_1980_2020
Y = rainfall_jeju
# %% 
plt.scatter(X, Y, alpha=0.5)
plt.xlabel('jeju sst')
plt.ylabel('jeju rainfall')
plt.show()
# %%
cov = np.cov(X, Y)[0, 1]
# %%
corr = np.corrcoef(X, Y)[0, 1]
print(corr)
# %%
pearsonr = stats.pearsonr(X, Y)
pvalue = pearsonr.pvalue
#%%

formatted_value = f"{pvalue:.40f}"

# 불필요한 소수점 자릿수를 제거하기 위해 문자열을 조정
formatted_value = formatted_value.rstrip('0').rstrip('.')

print(formatted_value)
print(pearsonr)
# %%

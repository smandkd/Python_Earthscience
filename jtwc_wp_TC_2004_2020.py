#%%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import geopandas as gpd 
from geopandas import GeoDataFrame as gdf

import warnings; warnings.filterwarnings(action='ignore')

dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
filtered_dataset = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
jtwc_wp_data = filtered_dataset.where(filtered_dataset.usa_agency == b'jtwc_wp', drop=True)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.set_global()
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.gridlines()

ax.set_xticks([0, 60, 120, 180, -180, -120, -60], crs=crs.PlateCarree())
ax.set_yticks([-60, -30, 0, 30, 60], crs=crs.PlateCarree())

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.scatter(x=jtwc_wp_data['lon'][:], y=jtwc_wp_data['lat'][:], s=1, alpha=0.5, transform=crs.PlateCarree())
plt.savefig("jtwc_wp_TC.pdf")
plt.show()



# %%

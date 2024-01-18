import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature

dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
dataset

filtered_dataset = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)

jtwc_tc = filtered_dataset.where((filtered_dataset.usa_agency == b'jtwc_wp') & (filtered_dataset.usa_wind > 34), drop=True)

tc_groups = jtwc_tc.groupby('sid')
tc_1 = tc_groups[b'2010240N15142']

lmi = tc_1.usa_wind.max()
lmi_index = np.where(tc_1.usa_wind == lmi)[0][-1]
tc_1_lon = tc_1.lon[:lmi_index+1]
tc_1_lat = tc_1.lat[:lmi_index+1]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.set_global()
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.gridlines()
ax.plot(tc_1_lon, tc_1_lat, transform=crs.PlateCarree(), linewidth=1)
plt.savefig("2010kompasu.pdf")
plt.show()
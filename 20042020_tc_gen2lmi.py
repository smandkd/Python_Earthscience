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

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.set_global()
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.gridlines()

for sid, tc_group in tc_groups:
    lmi = tc_group.usa_wind.max()
    lmi_index = np.where(tc_group.usa_wind == lmi)[0][-1]
    lon = tc_group.lon[:lmi_index+1]
    lat = tc_group.lat[:lmi_index+1]
    ax.plot(lon, lat, transform=crs.PlateCarree(), linewidth=1)
    # plt.scatter(lon, lat, s=1, alpha=0.5, transform=crs.PlateCarree())
    print(lmi)
    print(lmi_index)
    print(lon)
    print(lat)


plt.savefig("gen_2_lmi.pdf")
plt.show()
#%% 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature 

dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
dataset
# %%

filtered_data = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)
filtered_data

# %%

JTWC_tc = filtered_data.where(filtered_data.usa_agency == b'jtwc_wp', drop=True)
JTWC_tc

# %%

TC_kompasu = JTWC_tc.where(JTWC_tc.sid == b'2016232N22153', drop=True)
TC_kompasu

# %%

gen_INDEX = np.where(TC_kompasu.usa_wind >= 34)[-1][0]
print(gen_INDEX)

# %%

lmi = TC_kompasu.usa_wind.max()
lmi_INDEX = np.where(TC_kompasu.usa_wind == lmi)[-1][-1]
lmi_INDEX
# %%

kompasu_GEN2LMI_lat = TC_kompasu.lat[0][gen_INDEX:lmi_INDEX+1]
kompasu_GEN2LMI_lon = TC_kompasu.lon[0][gen_INDEX:lmi_INDEX+1]
print(TC_kompasu.usa_wind[0][gen_INDEX:lmi_INDEX+1])
print(kompasu_GEN2LMI_lat)
print(kompasu_GEN2LMI_lon)
# %%

extent_lat_min = min(kompasu_GEN2LMI_lat)
extent_lat_max = max(kompasu_GEN2LMI_lat)
extent_lon_min = min(kompasu_GEN2LMI_lon)
extent_lon_max = max(kompasu_GEN2LMI_lon)
padding = 5
extent = [extent_lon_min-padding, extent_lon_max+padding,
          extent_lat_min-padding, extent_lat_max+padding]


# %%

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.gridlines(draw_labels=True)
ax.set_extent(extent, crs=crs.PlateCarree())
ax.plot(kompasu_GEN2LMI_lon, kompasu_GEN2LMI_lat,
        transform=crs.PlateCarree(), linewidth=1)

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig("2016KOMPASU.pdf")
plt.show()


# %%

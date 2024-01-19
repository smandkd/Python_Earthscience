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
print(JTWC_tc)

# %%

for i in range(0,418):
    tc = JTWC_tc.isel(storm=i)
    print(tc)
    

# %%

TC_namtheun = JTWC_tc.where(JTWC_tc.sid == b'2004206N20151', drop=True)
TC_namtheun.usa_wind

# %%

gen_INDEX = np.where(TC_namtheun.usa_wind >= 34)[-1][0]
print(gen_INDEX)

# %%

lmi = TC_namtheun.usa_wind.max()
lmi_INDEX = np.where(TC_namtheun.usa_wind == lmi)[-1][-1]
lmi_INDEX
# %%

namtheun_GEN2LMI_lat = TC_namtheun.lat[0][gen_INDEX:lmi_INDEX+1]
namtheun_GEN2LMI_lon = TC_namtheun.lon[0][gen_INDEX:lmi_INDEX+1]
print(TC_namtheun.usa_wind[0][gen_INDEX:lmi_INDEX+1])
print(namtheun_GEN2LMI_lat)
print(namtheun_GEN2LMI_lon)
# %%

extent_lat_min = min(namtheun_GEN2LMI_lat)
extent_lat_max = max(namtheun_GEN2LMI_lat)
extent_lon_min = min(namtheun_GEN2LMI_lon)
extent_lon_max = max(namtheun_GEN2LMI_lon)
padding = 20
extent = [extent_lon_min-padding, extent_lon_max+padding,
          extent_lat_min-padding, extent_lat_max+padding]


# %%

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.gridlines(draw_labels=True)
ax.set_extent(extent, crs=crs.PlateCarree())
ax.plot(namtheun_GEN2LMI_lon, namtheun_GEN2LMI_lat,
        transform=crs.PlateCarree(), linewidth=1)

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig("2004NAMTHEUN.pdf")
plt.show()


# %%

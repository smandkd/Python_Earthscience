#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature

dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')

# %%

filtered_data = dataset.where((dataset.season >= 2004) & (dataset.season <=2020), 
                              drop=True)
filtered_data

# %%

jtwc_DATA = filtered_data.where((filtered_data.usa_agency == b'jtwc_wp') & (filtered_data.usa_wind >= 34), 
                                drop=True)
jtwc_DATA

# %%
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.gridlines(draw_labels=True)

for i in range(0, 415):
    i_TC = jtwc_DATA.isel(storm=i)
    
    index_GEN = np.where(i_TC.usa_wind >= 34)[0][0]
    
    LMI = i_TC.usa_wind.max()
    index_LMI = np.where(i_TC.usa_wind == LMI)[-1][-1]
    
    i_LAT = i_TC.lat[index_GEN:index_LMI+1]
    i_LON = i_TC.lon[index_GEN:index_LMI+1]
    ax.plot(i_LON, i_LAT,
            transform=crs.PlateCarree(),
            linewidth=1)

plt.savefig("JTWC_TC_GEN2LMI.pdf")
plt.show()

# %%

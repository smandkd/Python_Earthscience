# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import numpy as np

import warnings; warnings.filterwarnings(action='ignore')

dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')
dataset

# %%

filtered_dataset = dataset.where((dataset.season >= 2004) & (dataset.season <= 2020), drop=True)

# %%

jtwc_tc = filtered_dataset.where((filtered_dataset.usa_agency == b'jtwc_wp'), drop=True)


# %%

tc_tingting = jtwc_tc.where(jtwc_tc.name == b'TINGTING', drop = True)

tc_tingting_gen = tc_tingting.where(tc_tingting.usa_wind > 34, drop=True)
print(tc_tingting_gen.usa_wind)
lmi = tc_tingting_gen.usa_wind.max()
lmi_index = np.where(tc_tingting_gen.usa_wind == lmi)[1][3]
print("lmi : ", lmi)
print("lmi_index : ", lmi_index)


# %%

tc_KOMPASU = jtwc_tc.where(jtwc_tc.name == b'KOMPASU', drop = True)
tc_KOMPASU = tc_KOMPASU.where(tc_KOMPASU.usa_wind > 34, drop=True)
print(tc_KOMPASU.usa_wind)
lmi = tc_KOMPASU.usa_wind.max()
lmi_index = np.where(tc_KOMPASU.usa_wind == lmi)[1][0]

print("lmi : ", lmi)
print("lmi_index : ", lmi_index)

# %%

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
plt.scatter(x=tc_tingting_gen['lon'][:], y=tc_tingting_gen['lat'][:], s=1, alpha=0.5, transform=crs.PlateCarree())
plt.savefig("TC_tingting.pdf")
plt.show()

# %%

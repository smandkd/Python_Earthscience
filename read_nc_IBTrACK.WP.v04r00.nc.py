#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as crs
import cartopy.feature as cfeature

from netCDF4 import Dataset as netcdf_dataset

import sys
#from itertools import combination
from datetime import datetime
import random
import warnings 

warnings.filterwarnings("ignore")

fname = '/home/tkdals/typhoon_exp_1/IBTrACS.WP.v04r00.nc'
dataset = netcdf_dataset(fname)

lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]

fig = plt.figure(figsize=(10, 10)) # (가로, 세로)

ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree(central_longitude=180))
ax.set_global()
# ax.set_extent([100, 180, 7, 60])
# ax.set_extent([200, 300, 0, 90])

ax.add_feature(cfeature.COASTLINE, edgecolor="tomato")
ax.add_feature(cfeature.BORDERS, edgecolor="tomato")
ax.gridlines()

plt.scatter(x=lon, y=lat, color="dodgerblue", s=1, alpha=0.5, transform=crs.PlateCarree())
print(lat)
plt.savefig("storm.pdf")
plt.show()
# %%
"""
[[16.5 16.537290573120117 16.597816467285156 ... -- -- --]
 [16.100000381469727 16.111507415771484 16.13030242919922 ... -- -- --]
 [14.800000190734863 14.895709991455078 15.03433609008789 ... -- -- --]
 ...
 [16.299999237060547 16.21179962158203 16.205869674682617 ... -- -- --]
 [7.199999809265137 7.3100786209106445 7.3000006675720215 ... -- -- --]
 [7.6999993324279785 7.33000373840332 7.099999904632568 ... -- -- --]]
 
 
"""

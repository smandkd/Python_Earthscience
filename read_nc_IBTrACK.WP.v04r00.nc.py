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

fname = '/home/tkdals/typhoon_exp_1/IBTrACS.last3years.v04r00.nc'
dataset = netcdf_dataset(fname)
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(1,1,1, projection=crs.Robinson())
ax.set_extent([-50, 30, 27, 82])

ax.add_feature(cfeature.COASTLINE, edgecolor="tomato")
ax.add_feature(cfeature.BORDERS, edgecolor="tomato")
ax.gridlines()

plt.scatter(x=lat, y=lon, color="dodgerblue", s=1, alpha=0.5, transform=crs.PlateCarree())

plt.show()
# %%

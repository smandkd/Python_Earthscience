#%%
import netCDF4 as nc

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature

with nc.Dataset('IBTrACS.WP.v04r00.nc') as dataset:
    n_data = dataset.variables['name'][:]
    lat_data = dataset.variables['lat'][:]
    lon_data = dataset.variables['lon'][:]

    names = []
    current_name = ''

    for byte_array in n_data:
        name_str = ''.join(byte.decode('utf-8') for byte in byte_array if byte != b'')
        
        if name_str:
            names.append(name_str)
            
    if 'MAEMI' in names:
        maemi_index = names.index('MAEMI')
        maemi_lat = dataset.variables['lat'][maemi_index][:]
        maemi_lon = dataset.variables['lon'][maemi_index][:]    
    else:
        print("There is no such name maemi.")
        
    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(1, 1, 1, projection=crs.Robinson())
    ax.set_extent([100, 180, 7, 60])
    ax.add_feature(cfeature.COASTLINE, edgecolor="tomato")
    ax.add_feature(cfeature.BORDERS, edgecolor="tomato")
    ax.gridlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    plt.scatter(x=maemi_lon, y=maemi_lat, color="black", s=1, alpha=0.5, transform=crs.PlateCarree())
    plt.savefig("MAEMI_PATH.pdf")
    plt.show()



# %%

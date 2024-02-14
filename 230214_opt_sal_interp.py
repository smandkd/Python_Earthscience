
#%%
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
# %%
sal_dataset = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/SALT/2004/SALT.1440x720x50.20040612.nc')
opt_dataset = xr.open_dataset('/home/data/ECCO2/3DAYMEAN/THETA/2004/THETA.1440x720x50.20040612.nc')

sal_dataset['LONGITUDE_T'] = sal_dataset['LONGITUDE_T'] - 180
opt_dataset['LONGITUDE_T'] = opt_dataset['LONGITUDE_T'] - 180
# %%

dataset = xr.open_dataset('/home/tkdals/homework_3/IBTrACS.WP.v04r00.nc')
new_dataset = dataset.where((dataset.season >= 2004) &
                              (dataset.season <= 2020) & 
                              (dataset.usa_agency == b'jtwc_wp') &
                              (dataset.usa_wind >= 34), drop=True)

tc_CHANTHU = new_dataset.where(new_dataset.sid == b'2004158N07142', drop=True)
indexes = np.where(tc_CHANTHU.usa_wind >= 34)
gen_INDEX = indexes[0][0] # 34
lmi_INDEX = np.where(tc_CHANTHU.usa_wind == tc_CHANTHU.usa_wind.max())[-1][0]
selected_data = tc_CHANTHU.isel(date_time=slice(gen_INDEX, lmi_INDEX+1), drop=True)
# %%
chanthu_20040612 = selected_data.isel(date_time=12)

tc_center_coord = np.array(
    np.meshgrid(chanthu_20040612.lat.data, chanthu_20040612.lon.data)
).T.reshape(-1, 2)

# %%
sal_lon = sal_dataset.LONGITUDE_T.data 
sal_lat = sal_dataset.LATITUDE_T.data 
opt_lon = opt_dataset.LONGITUDE_T.data 
opt_lat = opt_dataset.LATITUDE_T.data 

sal_coords = np.array(
    np.meshgrid(sal_lon, sal_lat)
).T.reshape(-1, 2)
sal_tree = cKDTree(sal_coords)

opt_coords = np.array(
    np.meshgrid(opt_lon, opt_lat)
).T.reshape(-1, 2)
opt_tree = cKDTree(opt_coords)

dis_thres = 200/111

opt_indices = opt_tree.query_ball_point(tc_center_coord[0], dis_thres)
opt_tc_center_nearby_coords = opt_coords[opt_indices]

sal_indices = sal_tree.query_ball_point(tc_center_coord[0], dis_thres)
sal_tc_center_nearby_coords = sal_coords[sal_indices]
# %%
sal_tc_center_nearby_coords[:, 1]
sal_tc_center_nearby_coords[:, 0]

opt_tc_center_nearby_coords[:, 1]
opt_tc_center_nearby_coords[:, 0]
# %%
sal_points_chanthu_040612 = sal_dataset.sel(
        LATITUDE_T=sal_tc_center_nearby_coords[:, 0],
        LONGITUDE_T=sal_tc_center_nearby_coords[:, 1], 
        drop=True, method="nearest"
        )

opt_points_chanthu_040612 = opt_dataset.sel(
        LATITUDE_T=opt_tc_center_nearby_coords[:, 0],
        LONGITUDE_T=opt_tc_center_nearby_coords[:, 1], 
        drop=True, method="nearest"
        )

# %%
opt_depth = opt_points_chanthu_040612['DEPTH_T']
new_opt_depth = np.arange(5, opt_depth[-1] + 1, 1)
interpolated_opt = opt_points_chanthu_040612.interp(DEPTH_T=new_opt_depth, method="linear")

sal_depth = sal_points_chanthu_040612['DEPTH_T']
new_sal_depth = np.arange(5, sal_depth[-1] + 1, 1)
interpolated_sal = sal_points_chanthu_040612.interp(DEPTH_T=new_sal_depth, method="linear")# %%
# %%

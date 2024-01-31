#%%
import pandas as pd 
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
# %%
dataset = xr.open_dataset('IBTrACS.WP.v04r00.nc')

# %%
new_dataset = dataset.where((dataset.season >= 2004) &
                              (dataset.season <= 2020) & 
                              (dataset.usa_agency == b'jtwc_wp') &
                              (dataset.usa_wind >= 34), drop=True)

new_dataset
# %%
stacked_data = new_dataset.stack(all_dims=('storm', 'date_time'))
stacked_data.where(stacked_data.sid == b'2014197N10137', drop=True)
# %%

selected_data = stacked_data[['sid', 'name', 'usa_wind', 'time', 'usa_lon', 'usa_lat']]
selected_data
# %%

clean_data = selected_data.dropna(dim='all_dims', how="all")
clean_data
# %%
all_tc_data = clean_data.to_dataframe()
all_tc_data.reset_index(drop=True)
all_tc_data.head(65)
# %%
def get_gen_index(ds):
    gen_idx = np.where(ds['usa_wind'] >= 34)[0]
    return xr.DataArray(gen_idx[0] if len(gen_idx) > 0 else np.nan)
    
gen_idx = clean_data.groupby('storm').map(get_gen_index)
gen_idx

def get_lmi_index(ds):
    lmi_idx = np.where(ds['usa_wind'] == ds['usa_wind'].max())[0]
    return xr.DataArray(lmi_idx[0] if len(lmi_idx) > 0 else np.nan)
lmi_idx = clean_data.groupby('storm').map(get_lmi_index)
lmi_idx
# %%
all_tc_data['time_substraction_3d'] = all_tc_data['time'] - pd.Timedelta(days = 3)
# %%

selected_data = []
for storm in all_tc_data['storm'].unique():
    storm_data = all_tc_data[all_tc_data['storm'] == storm]
    start_idx = gen_idx.sel(storm=storm).item()
    end_idx = lmi_idx.sel(storm=storm).item()
    if not np.isnan(start_idx) and not np.isnan(end_idx):
        selected_data.append(storm_data.iloc[start_idx:end_idx + 1])
        
all_tc_data = pd.concat(selected_data, ignore_index=True)

# %%
all_tc_data.head(50)
# %%

years = all_tc_data.time.dt.year.unique()
datasets = {a: xr.open_dataset(f'/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.{a}.nc') 
            for a in years.unique()}

# %%
datasets = {year: xr.open_dataset(f'/home/data/NOAA/OISST/v2.1/only_AVHRR/Daily/sst.day.mean.{year}.nc') 
            for year in years}

oisst_coords = np.array(
    np.meshgrid(datasets[years[0]].lat, datasets[years[0]].lon - 180)
    ).T.reshape(-1, 2)
oisst_tree = cKDTree(oisst_coords)

# %%

results = []

for index, row in all_tc_data.iterrows():
    data = datasets[row['time'].year]
    oisst_sst = data.sst.sel(time=row['time'].strftime('%Y-%m-%d')).data.flatten()
    
    indices = oisst_tree.query_ball_point([row['lat'], row['lon']], 200/111)
    sst = oisst_sst[indices]
    
    results.append({
        'time': row['time'],
        'name': row['name'],
        'sid': row['sid'],
        'mea_sst': np.nanmean(sst)
    })
    
result_fd = pd.DataFrame(results)
result_fd.to_csv('result_sst_mean.csv', index=False)

# %%

data = pd.read_csv('preTC_sst_mean.csv', sep='\t', index_col=0)

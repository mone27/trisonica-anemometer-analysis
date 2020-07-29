from wind_tools import *

# %%
fpath = Path("test_data/20180809-2030_WM_174605_com1.raw")

n_samples = 30 / 5

df = pd.read_csv(fpath, usecols=(0,2,3,4,6), names=['tstmp','u', 'v', 'w', 't'], header=None)

df.groupby('tstmp').count().query('u == 9').count()

df.groupby(pd.cut(df.index, n_samples)).mean()

wm1 = load_ep_cache(wm1_path).rename(columns=wind_comp_rename)

rot = rotate_u_v(df, -310)

rotg = rot.groupby(pd.cut(df.index, n_samples)).mean()

#
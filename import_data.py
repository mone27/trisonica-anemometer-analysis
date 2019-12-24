from wind_tools import *
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
# %load_ext autoreload
# %autoreload 2
# sns.set()
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# %% define constants
proc_dir = Path("/r/media/simone/Simone DATI/TRISONICA_DATA/Processed/")
trs1_path = proc_dir / "TRS1_ago_sept" / "eddypro_TRS_1_full_output_2019-12-20T103011_exp.csv"
trs2_cor_path = proc_dir / "TRS2_cor" / "eddypro_TRS_2_cor_full_output_2019-12-21T140050_exp.csv"
wm1_path = proc_dir / "WM1_ago_sept" / "eddypro_WM1_full_output_2019-12-20T105458_exp.csv"
wm2_path = proc_dir / "WM1_ago_sept" / "eddypro_WM2_full_output_2019-12-20T124352_exp.csv"
cache_dir = Path("./processed_data")
period= pd.date_range('2018-08-10', '2018-08-13') # after 14 problems with TRS2 data

# %% load and filter datasets
wm1, wm2, trs1, trs2 = map(lambda p: load_ep(p).loc[period].loc[wind_cols].rename(wind_comp_rename),
                            [wm1_path, wm2_path, trs1_path, trs2_cor_path])

# %% hack: add plot metadata 
trs2.plot_info = {'label': 'TRS2 cor', 'color': "royalblue"}
wm1.plot_info = {'label': 'WM1', 'color': 'lightgreen'}
wm2.plot_info = {'label': 'WM2', 'color': 'darkgreen'}
trs1.plot_info = {'label': 'TRS1', 'color': 'orange'}
# trs2['v'] = -trs2.v
# invert u for TRS1: TODO fix it before EP processing
trs1.u = -trs1.u
# ??
trs2.v = -trs2.v
# ------------------------plots section----------------------
# %%
df1 = wm1-trs1
print(np.mean(df1[df1 > 0]))
print(np.mean(df1[df1 < 0]))
plot_dist_comp(wm1-trs1, ['u', 'v', 'w'])
plot_dist_comp(wm1-trs1, ['wind_speed'])
# %%
plot_components([wm1, wm2, trs1, trs2], ['u', 'v', 'w','wind_speed'])
plot_components([wm1, trs1], ['wind_speed', 'v'], vertical=False)
plot_components([wm1, trs1], ['v'])
plot_components([wm1, trs1], ['u'])
plot_components([wm1, trs1], ['w'])
plot_components([wm1, trs1], ['wind_speed'])
plot_components([wm1, trs1], ['v'], style='+')
# %%
# taking 30° range both north and south, wm1fa -> wm1 filtered angles
filt_angles = filter_by_wind_dir(wm1, 15)
wm1fa = wm1.loc[filt_angles]
trs2fa = trs2.loc[filt_angles]
trs2fa.plot_info = {'label': 'TRS2 uncor', 'color': "royalblue"}
wm1fa.plot_info = {'label': 'WM1', 'color': 'lightgreen'}
# %%
def fix_axes_trs2(df):
    """applies axes remap and rotations for a vertically mounted TRS, see notes.MD"""
    df = df.copy()
    df = rotate_u_v(df, angle=-np.pi/4)
    df.u, df.v, df.w = -df.u, -df.w, df.v
    return df
# %%
trs2r = trs2fa.pipe(fix_axes_trs2)
trs2r.plot_info = {'label': 'TRS2 cor', 'color': "royalblue"}
wm1 = wm1.pipe(add_angle_attack)
# %%
plot_components([wm1], ['angle_attack'])
plot_components([wm1fa, trs2r], ['u','v', 'w'], style='*-')
#%%
plot_components([wm1fa, trs2fa,], ['wind_speed', ], style=":o")
plot_components([wm1fa, trs2fa], ['w'], style=":o")
plot_components([wm1fa, trs2fa], ['wind_dir'], style='*')
# %% reference plot for TRS2 with fixed angles
fig1fa, ax1fa = plt.subplots(1,1)
wm1.wind_speed.plot(ax=ax1fa, **wm1.plot_info, style="+--")

# scaled_dir = wm1fa.wind_dir / 180
wm1fa.wind_speed.plot(ax=ax1fa, legend="Wind direction between 30° north or south", color="darkgreen", marker="o", linestyle="")
trs2fa.wind_speed.plot(ax=ax1fa, legend="Wind direction between 30° north or south", color="blue", marker="o", linestyle="")
trs2.wind_speed.plot(ax=ax1fa, **trs2.plot_info)
ax1fa.legend()
# %%
fig2fa, ax2fa = plt.subplots(1,1)
ax2fa.scatter(wm1fa.wind_speed, trs2fa.wind_speed)

# %%
fig = px.line(y=wm1.wind_speed, x=wm1.index)
fig.show()
# %%
nan_filter = ~(wm1.wind_speed.isna() | wm2.wind_speed.isna() | trs1.wind_speed.isna())
wm1 = wm1[nan_filter]
wm2 = wm2[nan_filter]
trs1 = trs1[nan_filter]
# %%
# plt.plot(trs2.wind_speed, label="TRS 2")
plt.plot(wm1.wind_speed, label="WM 1", color="lightgreen")
plt.plot(wm2.wind_speed, label="WM 2", color="darkgreen")

plt.plot(trs1.wind_speed, label="TRS 1", color="orangered")
# plt.bar(wm1.index, wm1.wind_speed - trs1.wind_speed)
plt.legend()
plt.tight_layout()
# %%

sns.distplot(wm1.wind_speed, color="lightgreen", label="WM 1")
sns.distplot(trs1.wind_speed, color="orangered", label="TRS 1")
plt.legend()
# %%
# plt.plot(trs1.u_rot, label="U trs1")
# plt.plot(trs2.u_rot, label="U trs2")
#
# plt.plot(trs1.v_rot, label="V trs1")
# plt.plot(trs2.v_rot, label="V trs2")
#
# plt.plot(trs1.w_rot, label="W trs1")
# plt.plot(trs2.w_rot, label="W trs2")
#
# plt.legend()

# %% windrose test

from windrose import WindroseAxes
import matplotlib.cm as cm
import numpy as np
ax1,ax2, ax3 = [ WindroseAxes.from_ax() for i  in range(3)]

ax1.bar(trs1.wind_dir, trs1.wind_speed, normed=True, opening=0.6, bins=8, label="Trisonica")
ax2.bar(wm1.wind_dir, wm1.wind_speed, normed=True, bins=8,label="Wind Master 1", opening=0.6)
ax3.bar(wm1.wind_dir, wm1.wind_speed, normed=True, bins=8,label="Wind Master 1", opening=0.6)
ax1.legend()
ax2.legend()
ax3.legend()
# # %% plotly express
# import plotly.express as px
#
# fig = px.bar_polar(trs1, r="wind_speed", theta="wind_dir")
# fig.show()
#
# # %% check vertical speed
# def add_dir_vert(df):
#     tot_hor = np.sqrt(df.u**2+df.v**2)
#     df['wind_dir_vert'] = np.arctan(df.w/tot_hor) * 180 / np.pi
#     return df
# def add_wind_u_w(df):
#     df['wind_speed_u_w'] = np.sqrt(df.u**2+df.w**2)
#
# add_dir_vert(wm1)
# wm1dv, trs1vd = [df.pipe(add_wind_u_w) for df in [wm1, trs1]]
#
# # wm1wv =
# # %%
# dfp = pd.DataFrame({"wm1_ws": wm1.wind_speed, "wm1_wd": wm1.wind_dir,
#                     "trs1_ws": trs1.wind_speed, "trs1_wd": trs1.wind_dir})

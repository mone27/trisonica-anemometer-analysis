import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from pandas import DataFrame

sns.set()
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
# %%
proc_dir = Path("/run/media/simone/Simone DATI/TRISONICA_DATA/Processed/")
trs1_path = proc_dir / "TRS1_ago_sept" / "eddypro_TRS_1_full_output_2019-12-20T103011_exp.csv"
wm1_path = proc_dir / "WM1_ago_sept" / "eddypro_WM1_full_output_2019-12-20T105458_exp.csv"
wm2_path = proc_dir / "WM1_ago_sept" / "eddypro_WM2_full_output_2019-12-20T124352_exp.csv"
cache_dir = Path("./processed_data")

# %% import ep full
def from_ep_full(file: Path) -> pd.DataFrame:
    return (pd.read_csv(file, skiprows=[0, 2], parse_dates=[['date', 'time']])\
            .set_index("date_time")
            .replace(-9999.0, np.NaN))
def filter_columns(df):
    return df.drop(columns=['DOY', 'filename', 'daytime', 'file_records', 'used_records','ET', 'water_vapor_density',
                            'e','specific_humidity', 'RH', 'VPD','Tdew', 'roll', 'bowen_ratio'])
def load_ep(file: Path) -> pd.DataFrame:
    """ load an edddypro full stat file, filtering columns and caching the result in an hdf5 file """
    hdf5_path = cache_dir / (file.name + ".h5")
    if hdf5_path.is_file():
        return pd.read_hdf(hdf5_path, key="df")
    else:
        df = from_ep_full(file).pipe(filter_columns)
        df.to_hdf(hdf5_path, key="df")
        return df
# %% load datasets
trs1 = load_ep(trs1_path)
wm1 = load_ep(wm1_path)
wm2 = load_ep(wm2_path)

# trs2 = load_ep("processed_data/eddypro_TRS_1_full_output_2019-12-19T184617_exp.csv")

# %%
dfp = pd.DataFrame({"wm1_ws": wm1.wind_speed, "wm1_wd": wm1.wind_dir,
                    "trs1_ws": trs1.wind_speed, "trs1_wd": trs1.wind_dir})
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
plt.plot(trs1.u_unrot, label="U trs1")
plt.plot(trs2.u_unrot, label="U trs2")

plt.plot(trs1.v_unrot, label="V trs1")
plt.plot(trs2.v_unrot, label="V trs2")

plt.plot(trs1.w_unrot, label="W trs1")
plt.plot(trs2.w_unrot, label="W trs2")

plt.legend()

# %% windrose test

from windrose import WindroseAxes
import matplotlib.cm as cm
import numpy as np
ax1 = WindroseAxes.from_ax()

ax1.bar(trs1.wind_dir, trs1.wind_speed, normed=True, opening=0.6,
       edgecolor='white', color='#ff000050', bins=8, label="Trisonica")
ax1.bar(trs1.wind_dir, wm1.wind_speed, normed=True, bins=8,
       edgecolor="blue", label="Wind Master", color="#00ffff50", opening=0.6)

ax1.legend()

# %% plotly express
import plotly.express as px

fig = px.bar_polar(trs1, r="wind_speed", theta="wind_dir")
fig.show()

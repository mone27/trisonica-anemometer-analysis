import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Iterable
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
# %% general funcs
def filt_dfs(dfs, filter):
    return [df.loc[filter] for df in dfs]
wind_cols = ['u_unrot','v_unrot', 'w_unrot', 'wind_speed', 'wind_dir']
wind_comp_rename = {'u_unrot': 'u', 'v_unrot': 'v', 'w_unrot': 'w'}
# %% plot helpers
def plot_components(dfs: Iterable[pd.DataFrame], cols=('u','v','w'), vertical=True, **kwargs):
    n_plts = (1,len(cols)); sharey = True; sharex = False
    if not vertical:n_plts=(len(cols), 1); sharey=False; sharex=True  #invert rows/columns
    fig, axes = plt.subplots(*n_plts, sharey=sharey, sharex=sharex)
    if not isinstance(axes, Iterable): axes = np.array([axes])  # if axes is no iterable  make it iterable
    for i, col in enumerate(cols):
        for df in dfs:
            df[col].plot(ax=axes[i], **(df.plot_info if hasattr(df,'plot_info') else {}), **kwargs)
        axes[i].set_title(col)
        axes[i].legend()
def plot_components_scatter(df1, df2, cols=('u','v','w'), vertical=True, linreg=True, **kwargs):
    n_plts = (1,len(cols)); sharey = True; sharex = False
    if not vertical:n_plts=(len(cols), 1); sharey=False; sharex=True  #invert rows/columns
    fig, axes = plt.subplots(*n_plts)
    if not isinstance(axes, Iterable): axes = np.array([axes])  # if axes is no iterable  make it iterable
    for i, col in enumerate(cols):
        axes[i].scatter(df1[col], df2[col], **kwargs)
        if linreg:
            # do linear regression
            df1_x = np.expand_dims(df1[col].to_numpy(), -1)
            reg = LinearRegression().fit(df1_x, df2[col])  # adding empty dimension for sklearn
            pred_y = reg.predict(df1_x)
            # plot theorical line
            axes[i].plot(df1[col],df1[col], color='green', label="theory if trs1=wm1")
            # plot actual regression line
            axes[i].plot(df1[col], pred_y, color='red', label=f"linear regression")
        axes[i].set_title(col)
        axes[i].set_xlabel(df1.plot_info['label'] if hasattr(df1,'plot_info') else "")
        axes[i].set_ylabel(df2.plot_info['label'] if hasattr(df2,'plot_info') else "")
        axes[i].legend()

def plot_dist_comp(df, cols):
    fig, axes = plt.subplots(1,1)
    for c in cols:
        sns.distplot(df[c], norm_hist=True, ax=axes, label=c)
    axes.legend()
# %% wind functions
def wind_speed(w_comp):
    return np.sqrt(sum([c**2 for c in w_comp]))
def filter_by_wind_dir(df, ang):
    """ filters when wind_dir is between +/- angle (in degrees) both north and south"""
    w = df.wind_dir
    return (w > (360 - ang)) | (w < ang) | ((w > (180-ang)) & (w < (180+ang)))

def add_angle_attack(df):
    df = df.copy()
    w_hor = np.sqrt(df.u**2+df.v**2)
    df['angle_attack'] = np.arctan(df.w/w_hor) * 180 / np.pi
    return df

def add_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['wind_speed'] = np.sqrt(sum([df[c]**2 for c in 'uvw']))
    return df
def add_wind_dir(df: pd.DataFrame) -> pd.DataFrame:
    print("warning is not the same of EddyPro")
    df = df.copy()
    df['wind_dir'] = 360 - (np.arctan2(df.v, df.u) * 180 / np.pi)
    return df
def rotate_u_v(df, angle):
    """rotate the components u and v by an angle considering the u as the x-axis"""
    df = df.copy()
    df.u = df.u*np.cos(angle) - df.v*np.sin(angle)
    df.v = df.u*np.sin(angle) + df.v*np.cos(angle)
    return df
# %% dataset loading from EddyPro
def from_ep_full(file: Path) -> pd.DataFrame:
    return (pd.read_csv(file, skiprows=[0, 2], parse_dates=[['date', 'time']])\
            .set_index("date_time")
            .replace(-9999.0, np.NaN))
def drop_empty_cols(df):
    return df.drop(columns=['DOY', 'filename', 'daytime', 'file_records', 'used_records','ET', 'water_vapor_density',
                            'e','specific_humidity', 'RH', 'VPD','Tdew', 'roll', 'bowen_ratio'])
def load_ep_cache(file: Path, cache_dir=Path(".")) -> pd.DataFrame:
    """ load an edddypro full stat file, filtering columns and caching the result in an hdf5 file """
    hdf5_path = cache_dir / (file.name + ".h5")
    if hdf5_path.is_file():
        return pd.read_hdf(hdf5_path, key="df")
    else:
        df = from_ep_full(file).pipe(drop_empty_cols)
        df.to_hdf(hdf5_path, key="df")
        return df

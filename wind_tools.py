import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Iterable
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.constants import convert_temperature
# %% general funcs

def test_close(a,b): assert np.allclose(a,b)

def get_ax(nrows=1, ncols=1): return plt.subplots(nrows, ncols)[1]

def mbe(a,b): return (a-b).abs().mean()
def mse(x, y): return (x-y).pow(2).mean()

def filt_dfs(dfs, filter):
    return [df.loc[filter] for df in dfs]

wind_cols = ['u_unrot','v_unrot', 'w_unrot', 'wind_speed', 'wind_dir']
wind_comp_rename = {'u_unrot': 'u', 'v_unrot': 'v', 'w_unrot': 'w'}

def c2k(t): return convert_temperature(t, 'Celsius', 'Kelvin')

def cart2pol(x, y):
    theta = np.arctan2(y,x)
    r = np.sqrt(x**2+y**2)
    return theta, r


def pol2cart(theta, r):
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return x, y

# %% plot helpers
def plot_components(dfs: Iterable[pd.DataFrame], cols=('u','v','w'), vertical=True, **kwargs):
    """for each component does a line plot with """
    n_plts = (1,len(cols)); sharey = True; sharex = False
    if not vertical:n_plts=(len(cols), 1); sharey=False; sharex=True  #invert rows/columns
    fig, axes = plt.subplots(*n_plts, sharey=sharey, sharex=sharex)
    if not isinstance(axes, Iterable): axes = np.array([axes])  # if axes is no iterable  make it iterable
    for i, col in enumerate(cols):
        for df in dfs:
            df[col].plot(ax=axes[i], **(df.plot_info if hasattr(df,'plot_info') else {}), **kwargs)
        axes[i].set_title(col)
        axes[i].legend()


def plot_components_scatter(dfs, cols=('u','v','w'), vertical=True, linreg=True, title=None, figsize=(6,5),**kwargs):
    try:
        df1, df2 = dfs
    except ValueError:
        raise("Need to pass exactly two dataframes for scatter plot")

    n_plts = (1,len(cols)); sharey = True; sharex = False

    if not vertical:n_plts=(len(cols), 1); sharey=False; sharex=True  #invert rows/columns
    fig, axes = plt.subplots(*n_plts, figsize=figsize)
    fig.suptitle(title)
    if not isinstance(axes, Iterable): axes = np.array([axes])  # if axes is no iterable  make it iterable
    for i, col in enumerate(cols):
        axes[i].scatter(df1[col], df2[col], **kwargs)
        df1_lbl = df1.plot_info['label'] if hasattr(df1,'plot_info') else ""
        df2_lbl = df2.plot_info['label'] if hasattr(df2,'plot_info') else ""
        if linreg:
            # do linear regression
            df1_x = np.expand_dims(df1[col].to_numpy(), -1)
            reg = LinearRegression().fit(df1_x, df2[col])  # adding empty dimension for sklearn
            pred_y = reg.predict(df1_x)
            # plot theoretical line
            axes[i].plot(df1[col],df1[col], color='green', label=f"theory if {df1_lbl}={df2_lbl}")
            # plot actual regression line
            axes[i].plot(df1[col], pred_y, color='red', label=f"linear regression")
        axes[i].set_title(col)
        axes[i].set_xlabel(df1_lbl)
        axes[i].set_ylabel(df2_lbl)
        axes[i].legend()


def plot_dist_comp(df, cols):
    fig, axes = plt.subplots(1,1)
    for c in cols:
        sns.distplot(df[c], norm_hist=True, ax=axes, label=c)
    axes.legend()


# %% wind functions
def wind_speed(df):
    return np.sqrt(sum([df[c]**2 for c in 'uvw']))


def filter_by_wind_dir(df, ang):
    """ filters when wind_dir is between +/- angle (in degrees) both north and south"""
    w = df.wind_dir
    return (w > (360 - ang)) | (w < ang) | ((w > (180-ang)) & (w < (180+ang)))


def add_angle_attack(df):
    df = df.copy()
    w_hor = np.sqrt(df.u**2+df.v**2)
    df['angle_attack'] = np.rad2deg(np.arctan2(df.w, w_hor))
    return df


def add_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['wind_speed'] = np.sqrt(df.u**2 + df.v**2 + df.w**2)
    return df

def get_wind_dir(df):
    """same behaviour of EP SingleWindDirection"""
    return (180 - np.rad2deg(np.arctan2(df.v, df.u))) % 360 
def fix_quadrant(wd): return (180 - wd) % 360


def add_wind_dir(wind, col_name='wind_dir'):
    """add wind EP wind dir to df"""
    wind = wind.copy()
    wind[col_name] = get_wind_dir(wind)
    return wind


def wind_speed_comp(df, w_comp):
    """calculate wind speed only with given components"""
    return np.sqrt(df.loc[:, list(w_comp)].pow(2).sum(axis=1)).copy()


# Warning need to check this is correct
def rotate_wind(df, ang):
    """rotate the u and v compoment, respectively x and y, of the wind by a given angle, using a rotation matrix.
    returns a datafram with u and v compoments"""
    return rotate_wind_comp(df, ang, ['u', 'v'])


# Warning need to check this is correct
def rotate_wind_comp(df, ang, comp):
    """rotate the give compomenst, respectively x and y, of the wind by a given angle, using a rotation matrix.
    returns a dataframe with the given compoments"""
    
    df = df.copy()
    ang = np.deg2rad(ang)
    rot_mat = np.array([[np.cos(ang), -np.sin(ang)],
                       [np.sin(ang), np.cos(ang)]])
    df[comp] = np.matmul(df[comp], rot_mat)
    return df

# note this is "math" wind dir not the meterological one (as used by EP)
# def add_wind_dir(df, wind_dir_name='wind_dir'):
#     df = df.copy()
#     df[wind_dir_name] = np.rad2deg(np.arctan2(df.v, df.u))
#     return df
    

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

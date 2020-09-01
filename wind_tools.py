import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Iterable
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.constants import convert_temperature

from scipy.spatial.transform import Rotation as R


import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import itertools
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# to mark that the angle must be in degrees and not radians
DegAng = int

# %% general funcs


def test_close(a,b): assert np.allclose(a,b)

def get_ax(nrows=1, ncols=1, **kwargs): return plt.subplots(nrows, ncols, **kwargs)[1]

def mbe(a,b): return (a-b).abs().mean()
def mse(x, y): return ((x-y)**2).mean()

def mod(ang: DegAng):
    """returns the modulo of 360"""
    return ang % 360

def filt_dfs(dfs, filter):
    return [df.loc[filter] for df in dfs]

def resample(dfs, rule):
    """shortcut to apply resample to a list of dfs, calc mean by default"""
    return list(map(lambda x: x.resample(rule).mean(), dfs))

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


# def rotate_ang(data, ang: DegAng):
#     """naive (but working) approach to rotate the u and v componets by given angles"""
#     wind_dir, wind_speed = cart2pol(data[:, 0], data[:, 1])
#     wind_dir += np.deg2rad(ang)
#     return np.column_stack(pol2cart(wind_dir, wind_speed))

def add_hor_wind_speed(df):
    df['wind_speed_hor'] = wind_speed_comp(df, ['u', 'v'])
    return df

def add_time_of_day(df):
    """Add a categorical column 'time_day' formatted and hour:minutes """
    df = df.copy()
    df['time_day'] = pd.Categorical(df.index.strftime("%H:%M"))
    return df

def add_wind_dir_binned(df, bins=8):
    df = df.copy()
    df['wind_dir_binned'] = pd.cut(df.wind_dir, bins=np.arange(0, 361, 360/bins))
    return df





# %% plot helpers
def plot_components(dfs: Iterable[pd.DataFrame], cols=('u','v','w'), vertical=False, plot_info=[], ax=None, **kwargs):
    """for each component does a line plot with """
    n_plts = (1,len(cols)); sharey = True; sharex = False
    if not vertical:n_plts=(len(cols), 1); sharey=False; sharex=True  #invert rows/columns
    
    if ax is not None: axes = ax # use user passed axes
    else:
        fig, axes = plt.subplots(*n_plts, sharey=sharey, sharex=sharex)
  
    if not isinstance(axes, Iterable): axes = np.array([axes])  # if axes is no iterable  make it iterable
    cols = list(cols)
    for i, col in enumerate(cols):
        for ii, df in enumerate(dfs):
            try:
                info = plot_info[ii]
            except IndexError:
                # if there is nothing try to get the plot info from the df otherwise fall back to empty one
                info = df.plot_info if hasattr(df,'plot_info') else {}
            
            df[col].plot(ax=axes[i], **(info), **kwargs)
        axes[i].set_title(col)
        axes[i].legend()
#         axes[i].grid()
    return axes


def plot_components_scatter(dfs, cols=('u','v','w'), vertical=True, linreg=True, title=None, figsize=None, plot_info=[], ax=None,**kwargs):
    try:
        df1, df2 = dfs
    except ValueError:
        raise("Need to pass exactly two dataframes for scatter plot")

    n_plts = (1,len(cols)); sharey = True; sharex = False

    if not vertical:n_plts=(len(cols), 1); sharey=False; sharex=True  #invert rows/columns
    
    if ax is not None: axes = ax # use user passed axes
    else:
        fig, axes = plt.subplots(*n_plts, figsize=figsize)
        fig.suptitle(title)
        
    if not isinstance(axes, Iterable): axes = np.array([axes])  # if axes is no iterable  make it iterable
    for i, col in enumerate(cols):
        axes[i].scatter(df1[col], df2[col], **kwargs)
        
        try:
            df1_lbl = plot_info[0]['label']
            df2_lbl = plot_info[1]['label']
        except IndexError:
            # if there is nothing try to get the plot info from the df otherwise fall back to empty one
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


def plot_dist_comp(dfs, cols='uvw', **kwargs):
    fig, axes = plt.subplots(1,1)
    for df in dfs:
        for c in cols:
            sns.distplot(df[c], norm_hist=True, ax=axes, label=c, **kwargs)
    axes.legend()


# %% wind functions
def wind_speed(df):
    return np.sqrt(sum([df[c]**2 for c in 'uvw']))


def filter_by_wind_ns_dir(df, ang):
    """ filters when wind_dir is between +/- angle (in degrees) both north and south"""
    w = filter_by_wind_dir(df, 360, ang)


def filter_by_wind_dir(df, start_ang: DegAng, range_ang: DegAng, both_dirs=True):
    """create a boolean filter where there direction is start_ang +/- range_ang.
     If both dirs considers (True also start_ang-180) +/- range_ang"""
    return filter_by_wind_dir_single(df.wind_dir, start_ang, range_ang, both_dirs)

def filter_by_wind_dir_single(wind_dir, start_ang: DegAng, range_ang: DegAng, both_dirs=True):
    """filter the wind_dir from the provide np array or Pandas Series"""
    filt = (wind_dir > mod((start_ang - range_ang))) & (wind_dir < mod(start_ang + range_ang))
    if both_dirs: filt = filt | ((wind_dir > mod(((start_ang-180) - range_ang))) & (wind_dir< mod((start_ang-180) + range_ang)))
    return filt

def add_angle_attack(df):
    df = df.copy()
    df['angle_attack'] = get_aoa(df.u, df.v, df.w)
    return df


def add_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['wind_speed'] = np.sqrt(df.u**2 + df.v**2 + df.w**2)
    return df

def get_wind_dir(u, v):
    """same behaviour of EP SingleWindDirection"""
    return mod((180 - np.rad2deg(np.arctan2(v, u))))

def get_aoa(u,v,w):
    w_hor = np.sqrt(u**2+v**2)
    return np.rad2deg(np.arctan2(w, w_hor))

def fix_quadrant(wd): return (180 - wd) % 360


def add_wind_dir(wind, col_name='wind_dir'):
    """add wind EP wind dir to df"""
    wind = wind.copy()
    wind[col_name] = get_wind_dir(wind.u, wind.v)
    return wind


def wind_speed_comp(df, w_comp):
    """calculate wind speed only with given components"""
    return np.sqrt(df.loc[:, list(w_comp)].pow(2).sum(axis=1)).copy()

def rotate_wind_hor_plane(wind, ang):
    
    wind_rot = R.from_euler('z', [ang], degrees=True)

    new_wind = wind_rot.apply(wind[['u', 'v', 'w']].copy().to_numpy())
    new_wind = pd.DataFrame(new_wind)
    new_wind.columns = list('uvw')
    new_wind.index = wind.index
    new_wind = new_wind.pipe(add_wind_speed).pipe(add_wind_dir).pipe(add_hor_wind_speed)
    
    return new_wind

# def rotate_wind_ang(df, ang):
#     df = df.copy()
#     wind_dir, wind_speed = cart2pol(df.u, df.v)
#     wind_dir += np.deg2rad(ang)
#     df.u, df.v = pol2cart(wind_dir, wind_speed)
#     return df

# Warning need to check this is correct
# def rotate_wind(df, ang):
#     """rotate the u and v compoment, respectively x and y, of the wind by a given angle, using a rotation matrix.
#     returns a datafram with u and v compoments"""
#     return rotate_wind_comp(df, ang, ['u', 'v'])


# # Warning need to check this is correct
# def rotate_wind_comp(df, ang, comp):
#     """rotate the give compomenst, respectively x and y, of the wind by a given angle, using a rotation matrix.
#     returns a dataframe with the given compoments"""
#     print("warning can be incorrrect")
#     df = df.copy()
#     ang = np.deg2rad(ang)
#     rot_mat = np.array([[np.cos(ang), -np.sin(ang)],
#                        [np.sin(ang), np.cos(ang)]])
#     df[comp] = np.matmul(df[comp], rot_mat)
#     return df

# note this is "math" wind dir not the meterological one (as used by EP)
# def add_wind_dir(df, wind_dir_name='wind_dir'):
#     df = df.copy()
#     df[wind_dir_name] = np.rad2deg(np.arctan2(df.v, df.u))
#     return df
 

    
## Fluxes

def add_H_raw_flux(df, interval):
    res = df.resample(interval)
    H = np.zeros(len(res))
    for i, (_, data) in enumerate(res):
        H[i] = data.w.cov(data.t)
    ret = res.mean()
    ret['H'] = H
    return ret

def add_Tau_raw_flux(df, interval):
    res = df.resample(interval)
    tau= np.zeros(len(res))
    for i, (_, data) in enumerate(res):
        tau[i] = np.sqrt(data.w.cov(data.u)**2 + data.w.cov(data.v)**2)
    ret = res.mean()
    ret['Tau'] = tau
    return ret

def add_raw_fluxes(df, interval):
    res = df.resample(interval)
    tau= np.zeros(len(res))
    H = np.zeros(len(res))
    for i, (_, data) in enumerate(res):
        tau[i] = np.sqrt(data.w.cov(data.u)**2 + data.w.cov(data.v)**2)
        H[i] = data.w.cov(data.t)
    ret = res.mean()
    ret['Tau'] = tau
    ret['H'] = H
    return ret.dropna()

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


def load_high_freq_data(f: Path, max_tol=.1) -> pd.DataFrame:
    """Load a proprocessed half an hour and returns an dataframe properly indexed"""
    df = pd.read_csv(f)
    if len(df) >= 18000: # half an hour at 10Hz
        df = df[:18000]
    else:
        empty_rows = pd.DataFrame([[np.nan] * len(df.columns)] * (18000 - len(df)), columns=df.columns)
        df = df.append(empty_rows)
    assert len(df) == 18000

    start_date = pd.to_datetime(f.name[:13])
    df.index = pd.TimedeltaIndex(df.index*100, unit='ms') + start_date
    
    
    if df.iloc[:, 0].isna().sum() < len(df)*max_tol:
        return df.interpolate()
    else:
        print(f"not enough data {f}")
        return df
    return df

#### vectors and 3d thing



# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

color_cycle = ['r', 'g', 'b', 'y', 'm']
def plot_vecs3d(vecs, colors=color_cycle, ax=None, lw=3):
    
    ax = ax or  plt.figure().add_subplot(111, projection='3d')    
    ax.set_xlim3d(min(vecs[:, 0].min(), -1), max(vecs[:, 0].max(), 1))
    ax.set_ylim3d(min(vecs[:, 1].min(), -1), max(vecs[:, 1].max(), 1))
    ax.set_zlim3d(min(vecs[:, 2].min(), -1), max(vecs[:, 2].max(), 1))
    ax.plot(0,0,0) #origin
    for vec, color in zip(vecs, itertools.cycle(colors)):
        a = Arrow3D([0, vec[0]], [0,vec[1]], [0, vec[2]], mutation_scale=20, 
                lw=lw, arrowstyle="-|>", color=color)
        ax.add_artist(a)
    plt.draw()
    return ax

axis_conv = {'x': 0, 'y': 1, 'z': 2}
def plot_vecs2d(vecs, plane='xy', colors=color_cycle, ax=None, lw=3):
    select = [axis_conv[plane[0]], axis_conv[plane[1]]]
    ax = ax or  get_ax()
    
    ax.set_xlim(min(vecs[:, select[0]].min(), -1) - .2, max(vecs[:, select[0]].max(), 1) +.2)
    ax.set_ylim(min(vecs[:, select[1]].min(), -1) - .2, max(vecs[:, select[1]].max(), 1) +.2)
    
    for vec, color in zip(vecs, itertools.cycle(colors)):
        ax.arrow(0,0, vec[select[0]], vec[select[1]], color=color, lw=lw)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    return ax
    

def plot_rotation_steps(v0, angs, euler='xyz', figsize=(18,16)):
    rot_ang = np.array([0,0,0])
    fig = plt.figure(figsize=figsize)
    vecs = [v0]
    
    for i in range(3):
        rot_ang[i] = angs[i]
        
        vecs.append(R.from_euler(euler, rot_ang, degrees=True).apply(v0))
        
        ax=fig.add_subplot(2,2,i+1, projection='3d')
        ax.set_title(f"adding a rotation on {euler[i]} of {angs[i]} rotation of {rot_ang}")
        plot_vecs3d(vecs[i+1], colors=['r', 'g', 'b'], ax=ax, lw=5)
        plot_vecs3d(vecs[i], colors=['fuchsia', 'yellow', 'cyan'], ax=ax, lw=2) # mark the old position
    
    ax = fig.add_subplot(224, projection='3d')
    ax.set_title("original vs final")
    plot_vecs3d(R.from_euler(euler, angs, degrees=True).apply(v0), colors=['r', 'g', 'b'], ax=ax, lw=5 )
    plot_vecs3d(v0, colors=['fuchsia', 'yellow', 'cyan'], ax=ax, lw=2) # plot origins as reference only on the last one
    

# TODO Rewrit this properly
def side_by_side(*objs, **kwds):
    ''' Une fonction print objects side by side '''
    from pandas.io.formats.printing import adjoin
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print(adjoin(space, *reprs))
        
        

v0 = np.array([[1,0,0], [0,1,0], [0,0,1]])

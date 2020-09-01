#!/usr/bin/env python

from pathlib import Path
import numpy as np
from multiprocessing import Pool
import logging as log
from dataclasses import dataclass
from functools import partial
from wind_tools import *
from scipy.spatial.transform import Rotation as R


u, v, w, t = 0, 1, 2, 3



# General utility func


def load(path, usecols, delimiter=None):
    return np.genfromtxt(path, usecols=usecols, delimiter=delimiter, invalid_raise=False)


def save(path, data, header="u,v,w,t"):
    np.savetxt(path, data, header=header, delimiter=',', fmt='%2.2f', comments='')
    log.info(f"saved {path}")

def should_process(path, replace):
    """checks if file does not exists or replace is True"""
    if replace or not path.exists():
        return True
    else: return False

def get_save_path(file, folder, name_suffix=""):
    return folder / f"{file.name[:-4]}{name_suffix}.csv"


def get_other_anem_name(f, other_suffix):
    """returns the file at the same fine but from a different anemometer"""
    return f.parent / (f.name[:13] + other_suffix)


def rotate(data, seq, angs):
    """take as input data where the 3 first compoments are u,v,w ; an euler sequence and euler angles.
     Returns rotated data"""
    rot = R.from_euler(seq, angs, degrees=True)
    rot_data = data.copy()  # to preserve other columns
    rot_data[:, [0, 1, 2]] = rot.apply(data[:, [0, 1, 2]])
    return rot_data


def replace_filt_dir(data, wind_dir, start_ang, range_ang, replace_value=-9999, both_dirs=True):

    filt = ~filter_by_wind_dir_single(wind_dir, start_ang, range_ang, both_dirs=both_dirs)

    length = min(len(data), len(filt))

    data[:length][filt[:length]] = replace_value

    return data

def replace_filt_aoa(data, aoa, range=30, replace_value=-9999):
    filt = abs(aoa) > range
    length = min(len(data), len(filt))
    data[:length][filt[:length]] = replace_value
    return data





def process(path, setg=None):
    save_path = get_save_path(path, setg.out_dir)
    if not should_process(save_path, setg.replace): return

    anem = load(path, *setg.load_cfg)
    anem = rotate(anem, *setg.rot)

    save(save_path, anem)

def process_filt_dir(path, setg=None):
    save_path = get_save_path(path, setg.out_dir, '_filt_dir')
    if not should_process(save_path, setg.replace): return
    
    anem = load(path, *setg.load_cfg)
    anem = rotate(anem, *setg.rot)

    wm_path = get_other_anem_name(path, setg.wm_name)
    wm = load(wm_path, *setg.wm_load_cfg)
    wm = rotate(wm, *setg.rot_wm)
    
    wd = get_wind_dir(wm[:,u], wm[:,v])
    anem_filt = replace_filt_dir(anem, wd, *setg.filt_ang)

    save(save_path, anem_filt)


def process_filt_aoa(path, setg=None):
    save_path = get_save_path(path, setg.out_dir, '_filt_aoa')
    if not should_process(save_path, setg.replace): return

    anem = load(path, *setg.load_cfg)
    anem = rotate(anem, *setg.rot)

    wm_path = get_other_anem_name(path, setg.wm_name)
    wm = load(wm_path, *setg.wm_load_cfg)
    wm = rotate(wm, *setg.rot_wm)

    aoa = get_aoa(wm[:, u], wm[:, v], wm[:, w])
    anem_filt = replace_filt_aoa(anem, aoa, *setg.filt_ang)

    save(save_path, anem_filt)
    

# process_wm1_filtered("2020_data/data_field_v2_from_20208010/raw/20200810-1430_WM_174605_com1.raw")

### parallel runner
### specific configutation and func
@dataclass
class Settings:
    in_dir = Path("2020_data/data_field_v2_from_20208010/raw")
    out_dir = Path("2020_data/data_field_v2_from_20208010/preprocessed")
    load_cfg: list
    rot: list
    name: str
    replace = True
    wm_load_cfg = [(2, 3, 4, 6), ',']
    wm_name = "_WM_174605_com1.raw"
    rot_wm = ['z', [50]]
    filt_ang = [250, 30]


setg_m506 = Settings(load_cfg=[(10, 12, 14, 16)], rot=['z', [-90]], name="*_TRS_M00506_com3.raw")
setg_m507 = Settings(load_cfg=[(10, 12, 14, 16)], rot=['XYZ', [90, -250, 135]], name="*_TRS_M00507_com2.raw")
setg_wm1 = Settings(load_cfg=[(2, 3, 4, 6), ','], rot=['z', [50]], name="*_WM_174605_com1.raw")

to_run = [(process, setg_m506 ),
          (process, setg_m507 ),
          (process, setg_wm1),
          (process_filt_aoa, setg_m506 ),
          (process_filt_aoa, setg_m507 ),
          (process_filt_aoa, setg_wm1),
          (process_filt_dir, setg_m506 ),
          (process_filt_dir, setg_m507 ),
          (process_filt_dir, setg_wm1),
          ]


def runner(to_run, parallel=True):
    with Pool() as p:
        for func, setg in to_run:

            print(f"{func.__name__} for {setg.name}")
            if parallel:
                p.map(partial(func, setg=setg), setg.in_dir.glob(setg.name))
            else:
                list(map(func, setg.in_dir.glob(setg.name)))


def main(log_level=log.INFO, parallel=True):
    log.basicConfig(level=log_level)
    runner(to_run, parallel)


if __name__ == '__main__':
    main()


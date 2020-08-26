#!/usr/bin/env python

from pathlib import Path
import numpy as np
from multiprocessing import Pool
import logging as log
from dataclasses import dataclass

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


def replace_filtered_wind_dir(data, wind_dir, start_ang, range_ang, replace_value=-9999, both_dirs=True):

    filt = ~filter_by_wind_dir_single(wind_dir, start_ang, range_ang, both_dirs=both_dirs)

    length = min(len(data), len(filt))

    data[:length][filt[:length]] = replace_value

    return data

def replace_filt_aoa(data, aoa, range=30, replace_value=-9999):
    filt = aoa.abs() < range
    length = min(len(data), len(filt))
    data[:length][filt[:length]] = replace_value
    return data


### specific configutation and func
@dataclass
class Settings:
    in_dir = Path("2020_data/data_field_v2_from_20208010/raw")
    out_dir = Path("2020_data/data_field_v2_from_20208010/preprocessed")

    trs_load_cfg = [(10, 12, 14, 16)]
    wm_load_cfg = [(2, 3, 4, 6), ',']

    rot_m6 = ['z', [-90]]
    rot_m7 = ['XYZ', [90, -250, 135]]
    rot_wm = ['z', [50]]
    filt_ang = [250, 30]

    replace = True

setg = Settings()


def process_m506(m6_path):
    save_path = get_save_path(m6_path, setg.out_dir)
    if not should_process(save_path, setg.replace): return

    m6 = load(m6_path, *setg.trs_load_cfg)
    m6 = rotate(m6, *setg.rot_m6)

    save(save_path, m6)


def process_m507(m7_path):
    save_path = get_save_path(m7_path, setg.out_dir)
    if not should_process(save_path, setg.replace): return

    m7 = load(m7_path, *setg.trs_load_cfg)
    m7 = rotate(m7, *setg.rot_m7)

    save(save_path, m7)


def process_wm(wm_path):
    save_path = get_save_path(wm_path, setg.out_dir)
    if not should_process(save_path, setg.replace): return

    wm = load(wm_path, *setg.wm_load_cfg)
    wm = rotate(wm, *setg.rot_wm)

    save(save_path, wm)


def process_wm1_filtered(wm_path):

    wm = load(wm_path, *setg.wm_load_cfg)
    wm = rotate(wm, *setg.rot_wm)

    wd = get_wind_dir(wm[:,0], wm[:,1])

    wm_filt = replace_filtered_wind_dir(wm, wd, *setg.filt_ang)

    save_path = get_save_path(wm_path, setg.out_dir, '_filt_dir')
    save(save_path, wm_filt)

def process_filt_dir(path):
    save_path = get_save_path(path, setg.out_dir, '_filt_dir')
    if not should_process(save_path, setg.replace): return
    
    anem = load(path, *setg.trs_load_cfg)
    anem = rotate(anem, *setg.rot_anem)

    wm_path = get_other_anem_name(path, '_WM_174605_com1.raw')    
    wm = load(wm_path, *setg.wm_load_cfg)
    wm = rotate(wm, *setg.rot_wm)
    
    wd = get_wind_dir(wm[:,0], wm[:,1])
    anem_filt = replace_filtered_wind_dir(anem, wd, *setg.filt_ang)

    save(save_path, anem_filt)


def process_filt_aoa(path):
    save_path = get_save_path(path, setg.out_dir, '_filt_dir')
    if not should_process(save_path, setg.replace): return

    anem = load(path, *setg.trs_load_cfg)
    anem = rotate(anem, *setg.rot_anem)

    wm_path = get_other_anem_name(path, '_WM_174605_com1.raw')
    wm = load(wm_path, *setg.wm_load_cfg)
    wm = rotate(wm, *setg.rot_wm)

    aoa = get_aoa(wm[:, u], wm[:, v], wm[:, w])
    anem_filt = replace_filtered_wind_dir(anem, aoa, *setg.filt_ang)

    save(save_path, anem_filt)
    

# process_wm1_filtered("2020_data/data_field_v2_from_20208010/raw/20200810-1430_WM_174605_com1.raw")

### parallel runner

to_run = [(process_m506, "*_TRS_M00506_com3.raw"),
          (process_m507, "*_TRS_M00507_com2.raw"),
          (process_wm, "*_WM_174605_com1.raw"),
          # (process_wm1_filtered, "*_WM_174605_com1.raw"),
          # (process_m507_filtered, "*_TRS_M00507_com2.raw"),
          ]


def runner(to_run, parallel=True):
    with Pool() as p:
        for func, glob in to_run:

            print(f"processing {func.__name__}")
            if parallel:
                p.map(func, setg.in_dir.glob(glob))
            else:
                list(map(func, setg.in_dir.glob(glob)))


def main(log_level=log.INFO, parallel=True):
    log.basicConfig(level=log_level)
    runner(to_run, parallel)


if __name__ == '__main__':
    main()


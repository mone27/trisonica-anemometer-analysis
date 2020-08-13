#!/usr/bin/env python

from pathlib import Path
import numpy as np
from multiprocessing import Pool
import logging as log

from wind_tools import *
from scipy.spatial.transform import Rotation as R

log.basicConfig(level=log.INFO)  # uncomment to see messages from extract



# General utility func


def load(path, usecols, delimiter=None):
    return np.genfromtxt(path, usecols=usecols, delimiter=delimiter, invalid_raise=False)


def save(path, data, replace=True, header="u,v,w,t"):
    if replace or not path.exists():
        np.savetxt(path, data, header="u,v,w,t", delimiter=',', fmt='%2.2f', comments='')
        log.info("saved", path)
    else:
        log.debug("skipped", path)


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


def replace_filtered_wind_dir(data, wind_dir, start_ang, range_ang, replace_value = -9999, both_dirs=True):

    filt = ~filter_by_wind_dir(wind_dir, start_ang, range_ang, both_dirs=both_dirs)

    length = min(len(data), len(filt))

    data[:length][filt[:length]] = replace_value

    return data



### specific configutation and func


in_dir = Path("2020_data/data_field_v2_from_20208010/raw")
out_dir = Path("2020_data/data_field_v2_from_20208010/preprocessed")

trs_load_cfg = [(10, 12, 14, 16)]

rot_m6 = ['z', [-90]]
rot_m7 = ['XYZ', [90, -250, 135]]
rot_wm = ['z', [50]]

u, v, w, t = 0, 1, 2, 3

replace = True


def make_mytrs(m6_path):
    # path is m6 path

    m7_path = get_other_anem_name(m6_path, '_TRS_M00507_com2.raw')

    m6 = load(m6_path, *trs_load_cfg)
    m7 = load(m7_path, *trs_load_cfg)

    m6 = rotate(m6, *rot_m6)
    m7 = rotate(m7, *rot_m7)

    length = min(len(m6), len(m7))  # can be slight different in length

    # Sonic temp taken from the m7 because it looked it was better .... maybe should make a mean
    mytrs = np.column_stack([m6[:length, u], m6[:length, v], m7[:length, w], m7[:length, t]])

    save_name = get_other_anem_name(m6_path, "_mytrs.csv")
    save_path = get_save_path(save_name, out_dir)

    save(save_path, mytrs)


def process_m506(m6_path):
    m6 = load(m6_path, *trs_load_cfg)
    m6 = rotate(m6, *rot_m6)
    save_path = get_save_path(m6_path, out_dir)
    save(save_path, m6, replace=replace)


def process_m507(m7_path):
    m7 = load(m7_path, *trs_load_cfg)
    m7 = rotate(m7, *rot_m7)
    save_path = get_save_path(m7_path, out_dir)
    save(save_path, m7, replace=replace)


def process_wm(wm_path):
    wm = load(wm_path, *wm_load_cfg)
    wm = rotate(wm, *rot_wm)
    save_path = get_save_path(wm_path, out_dir)
    save(save_path, wm, replace=replace)

def process_wm1_filtered(wm_path):
    wm = load(wm_path, *wm_load_cfg)
    wm = rotate(wm, *rot_wm)

    wd = get_wind_dir(wm[:,0], wm[:,1])

    wm_filt = replace_filtered_wind_dir()

    #work in progress

    save_path = get_save_path(wm_path, out_dir)
    save(save_path, wm, replace=replace)



### paralle runner


to_run = [(make_mytrs, "*_TRS_M00506_com3.raw"),
          (process_m506, "*_TRS_M00506_com3.raw"),
          (process_m507, "*_TRS_M00507_com2.raw"),
          (process_wm, "*_WM_174605_com1.raw")
          ]


def runner(to_run, parallel=True):
    with Pool() as p:
        for func, glob in to_run:

            print(f"processing {func.__name__}")
            if parallel:
                p.map(func, in_dir.glob(glob))
            else:
                list(map(func, in_dir.glob(glob)))


runner(to_run, True)

# fix axes for TRS1

from pathlib import Path
import numpy as np
name_re = r"*_TRS_M00162_com3.raw"
in_dir = Path("/run/media/simone/Simone DATI/TRISONICA_DATA/FIELD_DATA/agosto_settembra_all")
out_dir = Path("/run/media/simone/Simone DATI/TRISONICA_DATA/Processed/TRS1_all_data_fixed_axes")

def process_trs1(data):
    # fix TRS1 u ax by inverting it due to different coordinate system between trisonica and EP
    data[:, 0] = -1 * data[:, 0]
    return data

# trs = {'usecols': (5, 7, 9, 11),
#        'name_re': r"*_TRS_M00162_com3.raw",
#        'axes_trsf': process_trs1}
# wm = {'usecols': ()}


def process_trs1(data):
    # fix TRS1 u ax by inverting it due to different coordinate system between trisonica and EP
    data[:, 0] = -1 * data[:, 0]
    return data

def extract(f):
    data = np.genfromtxt(f, usecols=(5, 7, 9, 11))
    out_name = out_dir / f"{f.name[:-4]}_fixed_axes.csv"
    data = process_trs1(data)
    np.savetxt(out_name, data, header="u, v, w, t", delimiter=',', fmt='%2.2f')

def main():
    # possible optmization can use multithreads
    for f in in_dir.glob(name_re):
        extract(f)


if __name__ == '__main__':
    main()


# %%

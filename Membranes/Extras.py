#COMMONLY EMPLOYED, GENERIC FUNCTION. THIS MAY NOT BE UP TO DATE

import numpy as np

def center_bins(bins):
    return (bins[1:] + bins[:-1])/2

def ps_frame(ps, dt):
    return int(ps/dt)

def frame_ps(frame, dt):
    return frame * dt

def read_text_file(fname):
    f = open(fname)
    fl = f.readlines()
    f.close()
    clean = []
    for line in fl:
        if "#" not in line and "@" not in line:
            clean.append(line.split())
    clean = np.array(clean, dtype = 'float')
    return clean

def root_mean_squared_error(x1, x2):
    r = np.sqrt(np.sum(np.power(x1 - x2, 2))/len(x1))
    return r

def local_minima(x, y):
    dy = y[1:] - y[:-1]
    ndx_minima = np.where(np.logical_and(dy[1:] > 0, dy[:-1] < 0))[0] + 1
    return ndx_minima

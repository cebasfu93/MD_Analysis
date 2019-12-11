#CONTAINS GENERIC, OFTEN USED FUNCTIONS

import numpy as np


def cartesian_to_spherical(xyz):
    #r = pts[:,0] > 0
    #theta = pts[:,1] [0,pi]
    #phi = pts[:,2] [0, 2*pi]
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2])
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0]) #makes phi [-pi, pi]
    ptsnew[:,2] = (2*np.pi + ptsnew[:,2])%(2*np.pi)
    return ptsnew

def angle(p1, p2, p3):
    p21 = p2 - p1
    p23 = p2 - p3
    theta = np.acos(np.dot(p21, p23)/(np.linalg.norm(p21)*np.linalg.norm(p23)))
    return theta

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

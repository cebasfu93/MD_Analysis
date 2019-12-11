#Nanoparticle must be centered

#CALCULATES NUMBER DENSITY OF CERTAIN GROUPS AS A FUNCTION OF THE AZIMUTHAL ANGLE PHI (0,2PI) AND THE POLAR ANGLE THETA (0,PI) 
XTC = "NP18-53_PRO1_FIX.xtc" #D has the NP centered
TPR = "NP18-53_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("name AU AUS AUL"),
"heads"       : U.select_atoms("resname L18 and name C9"),
"NA"       : U.select_atoms("name NA"),
"CL"       : U.select_atoms("name CL"),
"OW"         : U.select_atoms("resname SOL and name OW")
}

props_solid = {
'ref'       : "all_gold",
'targets'    : ["heads", "NA", "CL", "OW"],
'start_ps'  : 25000,
'stop_ps'   : 100000,
'd_max'     : 22.5,
'phi_bins'  : 80,
'theta_bins': 40,
'dt'        : 100
}

def calc_theta(xyz):
    thetas = np.arccos(xyz[:,2]/np.linalg.norm(xyz, axis=1))
    return thetas

def calc_phi(xyz):
    phis = np.arctan2(xyz[:,1], xyz[:,0])
    return phis

def calculate_solid_angle_density(props):
    N_frames = 0
    THETAS = np.linspace(0, np.pi, props['theta_bins']+1)
    THETASc = center_bins(THETAS)
    dth = THETAS[1] - THETAS[0]
    PHIS = np.linspace(0, 2*np.pi, props['phi_bins']+1)
    PHISc = center_bins(PHIS)
    dph = PHIS[1] - PHIS[0]
    MESH_THETA, MESH_PHI = np.meshgrid(THETAS, PHIS)
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))

    results = {}
    for target in props['targets']:
        g_target = sel[target]
        print("Current target: {}".format(target))
        space = np.zeros((props['phi_bins'], props['theta_bins']))
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
                N_frames += 1
                print("Time -> {:.1f} ps".format(ts.time))
                x_target = np.subtract(g_target.positions, g_ref.center_of_mass())
                dists = np.linalg.norm(x_target, axis = 1)
                x_target = x_target[dists <= props['d_max']]
                th_target = calc_theta(x_target)
                ph_target = calc_phi(x_target)
                for th, ph in zip(th_target, ph_target):
                    space[int(ph//dph), int(th//dth)] += 1
            if ts.time >= props['stop_ps']:
                break

        space = space/N_frames
        N_counts = np.sum(space)
        print("Average counts: {:.2f}".format(N_counts))
        volumes = (MESH_PHI[1:,:-1] - MESH_PHI[:-1,:-1])*(props['d_max']/10)**3/3*(np.cos(MESH_THETA[:-1,:-1]) - np.cos(MESH_THETA[:-1,1:]))
        space = space/(volumes)
        results[target] = space

    return THETASc, PHISc, results

def write_solid(thetas, phis, res, props):
    f = open(NAME+"_denssolid.sfu", "w")
    f.write("#Number density of components\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    for target in props['targets']:
        f.write("#Phi (rad)/Theta (rad) ")
        for theta in thetas:
            f.write("{:>9.3f} ".format(theta))
        f.write("\n")

        for p, phi in enumerate(phis):
            f.write("{:<26.3f} ".format(phi))
            for t, theta in enumerate(thetas):
                f.write("{:>9.5f} ".format(res[target][p,t]))
            f.write("\n")
    f.close()

thetas, phis, results = calculate_solid_angle_density(props_solid)
write_solid(thetas, phis, results, props_solid)

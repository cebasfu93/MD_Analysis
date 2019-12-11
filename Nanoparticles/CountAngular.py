#CALCULATES THE NUMBER DENSITY OF GIVEN GROUPS AS A FUNCTION OF THE POLAR ANGLE THETA (0,PI)
XTC = "NP18-53_PRO1_FIX.xtc"
TPR = "NP18-53_PRO1.tpr"
NAME = XTC[:-8]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as axes3d
from Extras import *
from MDAnalysis import *
from physt import special
plt.rcParams["font.family"] = "Times New Roman"
z = 22

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("name AU AUS AUL"),
"C9"        : U.select_atoms("resname L18 and name C9"),
"NA"        : U.select_atoms("resname NA"),
"CL"        : U.select_atoms("resname CL"),
"SOL-OW"    : U.select_atoms("resname SOL and name OW"),
}

props_count = {
'ref'       : "all_gold",
'targets'   : ["C9", "NA", "CL", "SOL-OW"],
'start_ps'  : 25000, #Start of window in which calculate the histogram
'stop_ps'   : 100000,
'd_max'     : 22.5,
'dt'        : 10,
'bins'      : 60,
'res'       : 1024,
'plot'      : True
}

def count_angular(props):
    N_frames = 0
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))
    bins, all_dens = [], []
    for target in props['targets']:
        g_target = sel[target]
        print("Current target: {}".format(target))
        count = []
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
                N_frames += 1
                #print(ts.time)
                x_target = np.subtract(g_target.positions, g_ref.center_of_mass())
                dists = np.linalg.norm(x_target, axis = 1)
                count += list(x_target[dists <= props['d_max']].flatten())
            if ts.time >= props['stop_ps']:
                break
        count = np.array(count).reshape((len(count)//3, 3))
        h = special.spherical_histogram(count, theta_bins = props['bins'])
        h = h.projection('theta')
        counts = h.frequencies/N_frames
        N_counts = np.sum(counts)
        vols = 2.*np.pi/3*(props['d_max']/10)**3*(np.cos(h.numpy_bins[:-1]) - np.cos(h.numpy_bins[1:]))
        densities = np.divide(counts, vols)
        all_dens.append(densities)
        print("Average counts: {:.2f}".format(N_counts))

        if props['plot']:
            homo_dens = 3.*N_counts/(4*np.pi*(props_count['d_max']/10)**3)
            print(homo_dens)
            db = h.numpy_bins[1] - h.numpy_bins[0]

            x0, y0 = props_count['res']/2, props_count['res']/2
            v0 = np.array([x0, y0])
            space = np.full((props_count['res'],props_count['res']), homo_dens)
            thetas = []
            for y in range(props_count['res']):
                for x in range(props_count['res']):
                    v = np.array([x,y]) - v0
                    r = np.linalg.norm(v)
                    if r < props_count['res']/2 and r != 0:
                        th = np.arccos(v[1]/r)
                        thetas.append(th)
                        space[y,x] = densities[int((th%np.pi)//db)-1]

            f_pm = 0.4
            fig = plt.figure()
            ax = plt.axes()
            cax = ax.imshow(space, vmin=(1-f_pm)*homo_dens, vmax = (1+f_pm)*homo_dens, interpolation = 'bilinear', cmap='bwr')
            cbar = fig.colorbar(cax, ticks = [(1-f_pm)*homo_dens, homo_dens, (1+f_pm)*homo_dens])
            cbar.ax.set_yticklabels(['-{:.0f}%'.format(f_pm*100), 'Uniform', '+{:.0f}%'.format(f_pm*100)])
            cbar.ax.tick_params(labelsize = z)
            plt.show()

    all_dens = np.array(all_dens).T


    return h.numpy_bins[:-1], all_dens #h.numpy_bins go from 0 - pi

def write_count(bins, dens, props):
    f = open(NAME+"_densangular_tmp.sfu", "w")
    f.write("#Density (number nm-3)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Theta (rad)")
    for key in props['targets']:
        f.write("{:>9} ".format(key))
    f.write("\n")

    for i in range(len(bins)):
        f.write("{:<11.3f} ".format(bins[i]))
        for j in range(len(dens[0,:])):
            f.write("{:>9.3f} ".format(dens[i,j]))
        f.write("\n")
    f.close()

bins, densities = count_angular(props_count)
write_count(bins, densities, props_count)

"""    dens, bins, N_counts = count_angular(props_count, i)
    homo_dens = 3.*N_counts/(4*np.pi*(props_count['d_max']/10)**3)
    db = bins[1] - bins[0]

    x0, y0 = props_count['res']/2, props_count['res']/2
    v0 = np.array([x0, y0])
    space = np.full((props_count['res'],props_count['res']), homo_dens)
    thetas = []
    for y in range(props_count['res']):
        for x in range(props_count['res']):
            v = np.array([x,y]) - v0
            r = np.linalg.norm(v)
            if r < props_count['res']/2 and r != 0:
                th = np.arccos(v[1]/r)
                thetas.append(th)
                space[y,x] = dens[int(th//db)]

    f_min, f_max = 0.6, 1.4
    fig = plt.figure()
    ax = plt.axes()
    cax = ax.imshow(space, vmin=f_min*homo_dens, vmax = f_max*homo_dens, interpolation = 'bilinear', cmap='bwr')
    cbar = fig.colorbar(cax, ticks = [homo_dens*f_min, homo_dens, homo_dens*f_max])
    cbar.ax.set_yticklabels(['-40%', 'Uniform', '+40%'])
    cbar.ax.tick_params(labelsize = z)
    plt.show()"""

#CALCULATES THE AVERAGE AREA PER LIPID FOR EACH FRAME. IT CAN ALSO MAKE VORONOI PLOTS
#THIS SHOULD BE GENERALIZED TO GET THE XYZ COORDINATES AND THE AREA PER LIPID OF EACH LIPID

XTC = "POPC2-24_PRO1-2_FIX.xtc"
TPR = "POPC2-24_PRO1.tpr"
NAME = XTC[:-8]

from MDAnalysis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, voronoi_plot_2d
import multiprocessing

U = Universe(TPR, XTC)
sel = {
"P31" : U.select_atoms("name P31")
}

props_apl={
"ref"       : 'P31',
"start_ps"  : 25000,
"stop_ps"   : 100000,
"dt"        : 80,
"up"        : True,
"down"      : False,
"n_proc"    : 12,
}

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def apl_one_frame(coords_mem):
    #For paralelization, one of up/down must be set to true by default
    Z_mean = np.mean(coords_mem[:,2])
    upper = coords_mem[coords_mem[:,2] > Z_mean,:]
    lower = coords_mem[coords_mem[:,2] < Z_mean,:]
    if props_apl['up']:
        leaflet = upper
    elif props_apl['down']:
        leaflet = lower
    headx_max, headx_min = np.max(leaflet[:,0]), np.min(leaflet[:,0])
    heady_max, heady_min = np.max(leaflet[:,1]), np.min(leaflet[:,1])

    vor_leaf = Voronoi(leaflet[:,:2])
    points = []
    areas = []
    for region in vor_leaf.regions:
        if not -1 in region:
            polygon = [vor_leaf.vertices[i] for i in region]
            if polygon:
                x = np.array([])
                y = np.array([])
                for v in polygon:
                    x = np.append(x, v[0])
                    y = np.append(y, v[1])
                points.append(np.vstack((x,y)).T)
                areas.append(PolyArea(x, y))
    nice_points = []
    nice_areas = []
    for i in range(len(points)):
        if np.all(points[i][:,0] > headx_min) and np.all(points[i][:,0] < headx_max) and np.all(points[i][:,1] > heady_min) and np.all(points[i][:,1] < heady_max) :
            nice_points.append(points[i])
            nice_areas.append(areas[i])
    return nice_points, nice_areas, leaflet

def apl_over_time(props):
    apl_ave = []
    apl_std = []
    times = []
    all_coords = []
    g_ref = sel[props["ref"]]
    for ts in U.trajectory:
        if ts.time >= props["start_ps"] and ts.time%props["dt"]==0:
            times.append(ts.time)
            all_coords.append(g_ref.positions)
        elif ts.time >= props["stop_ps"]:
            break

    pool = multiprocessing.Pool(processes = props["n_proc"])
    apl = pool.map(apl_one_frame, all_coords)
    for i in range(len(apl)):
        apl_ave.append(np.mean(apl[i][1]))
        apl_std.append(np.std(apl[i][1]))

    times = np.array(times)
    apl_ave = np.array(apl_ave)/100 #100 A2 to nm2
    apl_std = np.array(apl_std)/100

    return times, apl_ave, apl_std

def write_apl(times, apl_mean, apl_std, props):
    f = open(NAME + "_apl.sfu", "w")
    f.write("#Average area per lipid (nm2)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Area per lipid (mean of all values +/- deviation of all values): {:.3f} +/- {:.3f} nm2\n".format(np.mean(apl_mean), np.mean(apl_std)))
    f.write("#Time (ps)\tMean apl (nm2)\tDeviation apl (nm2)\n")
    for i in range(len(apl_mean)):
        f.write("{:<9.3f} {:<9.4f} {:<9.4f}\n".format(times[i], apl_mean[i], apl_std[i]))
    f.close()

def plot_apl(points, areas, leaflet):
    fig = plt.figure()
    ax = plt.axes()
    cmap = cm.PRGn
    norm = Normalize(vmin=np.min(areas), vmax = np.max(areas))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    for i in range(len(points)):
        polygon = np.append(points[i], [points[i][0,:]], axis = 0)
        ax.plot(polygon[:,0], polygon[:,1], lw = 1, color = (0.4, 0.4, 0.4))
        ax.fill(points[i][:,0], points[i][:,1], color = cmap(norm(areas[i])), linewidth = 1, alpha = 0.3)
    ax.scatter(leaflet[:,0], leaflet[:,1], color = 'g', s = 6.0)
    sm._A = areas
    plt.colorbar(sm)
    plt.show()
    plt.close()


time, aplm, apls = apl_over_time(props_apl)
write_apl(time, aplm, apls, props_apl)

#CALCULATES THE AVERAGE AREA PER LIPID FOR EACH FRAME. IT CAN ALSO MAKE VORONOI PLOTS
#THIS SHOULD BE GENERALIZED TO GET THE XYZ COORDINATES AND THE AREA PER LIPID OF EACH LIPID

XTC = "POPC2-24_NPT_FIX.xtc"
TPR = "POPC2-24_NPT.tpr"
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
"start_ps"  : 0,
"stop_ps"   : 10000,
"dt"        : 10,
"up"        : True,
"down"      : False,
"n_proc"    : 8,
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
    good_pts = np.zeros(len(vor_leaf.points), dtype='bool')

    for i in range(len(vor_leaf.points)):
        region = vor_leaf.regions[vor_leaf.point_region[i]]
        if not -1 in region:
            polygon = [vor_leaf.vertices[j] for j in region]
            if polygon:
                x = np.array([])
                y = np.array([])
                for v in polygon:
                    x = np.append(x, v[0])
                    y = np.append(y, v[1])
                points = np.vstack((x,y)).T
                if np.all(points[:,0] > headx_min) and np.all(points[:,0] < headx_max) and np.all(points[:,1] > heady_min) and np.all(points[:,1] < heady_max):
                    good_pts[i] = True
                    areas.append(PolyArea(x, y))

    nice_coords = leaflet[good_pts]
    return nice_coords, areas

def apl_over_time(props):
    times = []
    all_coords = []
    out_coords = []
    out_apls = []

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
        out_coords.append(np.array(apl[i][0])/10) #A to nm
        out_apls.append(np.array(apl[i][1])/100) #A2 to nm2

    times = np.array(times)

    return times, out_coords, out_apls

def write_apl(times, coords, apls, props):
    f = open(NAME + "_apl.sfu", 'w')
    f.write("#Area per lipid of each lipid (lipids close to the edges are discarded for instability during the Voronoi analysis)")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#The following coordinates are those of the heads\n")
    f.write("#{:<8} {:<9} {:<9}\n".format("X (A)", "Y (A)", "APL"))

    for i in range(len(times)):
        f.write("#T -> {:<10.3f} ps\n".format(times[i]))
        for j in range(len(apls[i])):
            f.write("{:<9.3f} {:<9.3f} {:<9.3f}\n".format(coords[i][j,0], coords[i][j,1], apls[i][j]))
    f.close()

def plot_apl(points, areas, leaflet):
    #Deprecated
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
    
time, coords, apls = apl_over_time(props_apl)
write_apl(time, coords, apls, props_apl)

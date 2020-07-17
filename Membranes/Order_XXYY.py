"""
Calulates the order parameter for all the given atoms in the lipid chains (using the Ci-1 - Ci - Ci+1 vector).
It prints the XY coordinate of the phosphate headgroup followed by all the order parameters belonging to that lipid.
Useful for building bidimensional maps of the lipid order.
"""
XTC = "NP61-POPC6-46_PRO1_Pt_FIX.xtc"
TPR = "NP61-POPC6-46_Pt512.tpr"
NAME = XTC[:-8]

from MDAnalysis import *
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

U = Universe(TPR, XTC)
sel = {
'MEM' : U.select_atoms("resname POPC"),
'PO4' : U.select_atoms("name PO4"),
}

props_order_xxyy = {
'mem_group'   : 'MEM',
'ref_group'   : 'PO4',
'N_proc' : 12,
'UP'     : False,
'DOWN'  : True,
'chains' : [['PO4', 'GL1','C1A', 'D2A', 'C3A', 'C4A'],\
        ['PO4', 'GL2','C1B', 'C2B', 'C3B', 'C4B']],
'start_ps' : 0,
'stop_ps' : 1000000,
'dt'   : 10
}

def order_parameter(ang):
    return 1.5*(np.cos(ang)**2) -0.5

def angle_from_z(vs, updown):
    dot = vs[:,2]*updown
    ang = np.arccos(np.clip(dot, -1,1))
    return ang

def norm_vecs(vs):
    return vs/(np.array([np.linalg.norm(vs, axis=1)]).T)

def order_xxyy(props):
    updown = props['UP']-props['DOWN']
    g_mem = sel[props['mem_group']]
    g_ref = sel[props['ref_group']]
    mid_plane = g_ref.centroid()[2]
    leaf_mask = (g_ref.positions[:,2]-mid_plane)*(updown) >0
    #leaf_mask = np.zeros(len(g_ref.atoms), dtype='bool')
    #leaf_mask[0] = True
    a_ndxs = [[np.where(g_mem.atoms.names == name)[0][leaf_mask] for name in chain] for chain in props['chains']]

    times, heads, orders = [], {}, {}
    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time%props['dt']==0:
            if ts.time%100000 == 0:
                print(ts.time)
            times.append(ts.time)
            x0 = np.array([[g_mem.positions[atoms] for atoms in chain[:-2]] for chain in a_ndxs])
            x1 = np.array([[g_mem.positions[atoms] for atoms in chain[1:-1]] for chain in a_ndxs])
            x2 = np.array([[g_mem.positions[atoms] for atoms in chain[2:]] for chain in a_ndxs])
            ori_shape = x0.shape
            x0 = x0.reshape(-1, ori_shape[-1])
            x1 = x1.reshape(-1, ori_shape[-1])
            x2 = x2.reshape(-1, ori_shape[-1])
            zz = norm_vecs(x2-x0)
            #xx = norm_vecs(np.cross(zz, x2-x1))
            #yy = norm_vecs(np.cross(zz, xx))

            #angx = angle_from_z(xx, updown)
            #angy = angle_from_z(yy, updown)
            #ordx, ordy = order_parameter(angx), order_parameter(angy)
            #ord = (2*ordx + ordy)/3.
            ord = order_parameter(angle_from_z(zz, updown))
            new_shape = (ori_shape[0]*ori_shape[1], ori_shape[2])
            orders[ts.time] = ord.reshape(new_shape)
            heads[ts.time] = g_ref.positions[leaf_mask]/10. #10 A to nm
        elif ts.time > props['stop_ps']:
            break

    return times, heads, orders

def write_orders(times, heads, orders, props):
    right_names = np.array([[c for c in props['chains'][i][1:-1]] for i in range(len(props['chains']))]).flatten()
    f = open(NAME+"_order_xy.sfu", "w")
    f.write("#Order parameter from Sxx and Syy\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    vals = [v for v in orders.values()]
    print(np.min(vals), np.max(vals))
    f.write("#Order parameter (mean of all values +/- deviation of all values): {:.3f} +/- {:.3f} \n".format(np.mean(vals), np.std(vals)))
    f.write("#X (nm)  Y (nm)" + ("  {:>10}"*len(right_names)).format(*right_names) + "\n")
    for time in times:
        f.write("#T - > {:.1f} ps\n".format(time))
        for head, order in zip(heads[time], orders[time].T):
            f.write("{:<10.4f} {:<10.4f}".format(*head) + (" {:<10.4f}"*len(order)).format(*order) + "\n")
    f.close()
    """vals = np.array(vals)
    fig = plt.figure()
    ax = plt.axes()
    means = np.mean(vals, axis=(0,2))
    stds = np.std(vals, axis=(0,2))
    x = np.linspace(1,4,4)
    #ax.errorbar(x,means[:4], fmt='o-')
    #ax.errorbar(x,means[4:], fmt='o-')
    ax.errorbar(x,means[:4], yerr=stds[:4], fmt='o-')
    ax.errorbar(x,means[4:], yerr=stds[4:], fmt='o-')
    plt.show()"""


times, heads, orders = order_xxyy(props_order_xxyy)
write_orders(times, heads, orders, props_order_xxyy)

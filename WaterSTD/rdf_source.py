import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from tqdm import tqdm
from MDAnalysis import *

def rdf(U, props, sel):
    """
    Calculates radial distribution function with respect to the center of mass of a reference group.
    The RDF of the solvent does not converge to 1 because the non-negligible volume of the nanoparticle.
    """
    DT = U.trajectory[0].dt
    n_read = int(props['stop_ps'] - props['start_ps'])/DT + 1
    rdfs = {}
    rdfs_cum = {}
    R = np.linspace(props['r_range'][0], props['r_range'][1], props['nbins'])
    R_center = 0.5*(R[1:] + R[:-1])
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))
    for target in props['targets']:
        n_frames = 0
        print("Current target: {}".format(target))
        counts = np.zeros(len(R)-1)
        g_target = sel[target]
        for ts in tqdm(U.trajectory, total=n_read):
            if ts.time > props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                n_frames += 1
                x_ref = g_ref.center_of_mass()
                x_target = g_target.positions
                dx = np.subtract(x_target, x_ref)
                dists = np.linalg.norm(dx, axis = 1)
                dists = dists[dists<=props['r_range'][1]]
                counts += np.histogram(dists, bins = R)[0]
        n_target = np.sum(counts)/n_frames #Average particles counted in target group
        V_shells = (4*np.pi*(R[1:]**3-R[:-1]**3)/3)
        homo_dens = 3*n_target/(4*np.pi*R[-1]**3)
        counts /= n_frames #Averages number of target particles at each bin over time
        counts /= 1 #Divide on the number of reference particles (1 center of mass)
        counts /=  V_shells#Divide on shell volume
        counts /= homo_dens #Divide on homogeneous density
        rdfs[target] = counts
        integrand = 4*np.pi*np.multiply(R_center**2, counts)
        cumulative = homo_dens*cumtrapz(y=integrand, x=R_center, initial=0.0) #returns a shape with one element less
        rdfs_cum[target] = cumulative
    return R_center, rdfs, rdfs_cum

def pair_rdf(U, props, sel):
    """
    Calculates pair-wise distribution function between two groups.
    """
    DT = U.trajectory[0].dt
    n_read = int(props['stop_ps'] - props['start_ps'])/DT + 1
    rdfs = {}
    R = np.linspace(props['r_range'][0], props['r_range'][1], props['nbins'])
    R_center = 0.5*(R[1:] + R[:-1])
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))
    for target in props['targets']:
        n_frames = 0
        print("Current target: {}".format(target))
        counts = np.zeros(len(R)-1)
        g_target = sel[target]
        for ts in tqdm(U.trajectory, total=n_read):
            if ts.time > props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                n_frames += 1
                x_ref = g_ref.positions
                x_target = g_target.positions
                dists = cdist(x_ref, x_target)
                dists = dists[dists<=props['r_range'][1]]
                counts += np.histogram(dists, bins = R)[0]
        n_ref = g_ref.n_atoms
        n_target = np.sum(counts)/n_frames #Average particles counted in target group
        V_shells = (4*np.pi*(R[1:]**3-R[:-1]**3)/3)
        homo_dens = 3*n_target/(4*np.pi*R[-1]**3)
        counts /= n_frames #Averages number of target particles at each bin over time
        counts /= n_ref #Divide on the number of reference particles (1 center of mass)
        counts /=  V_shells#Divide on shell volume
        counts /= homo_dens #Divide on homogeneous density
        rdfs[target] = counts
    return R_center, rdfs

def write_rdf(space, rdf_dict, properties):
    f = open(NAME + "_rdf.sfu", 'w')
    values = []
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space(nm) ")
    for key, val in rdf_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)
    for i in range(len(space)):
        f.write("{:<8.3f} ".format(space[i]/10.)) #10 A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i]))
        f.write("\n")
    f.close()

def write_rdf_cum(space, rdf_dict, properties):
    f = open(NAME + "_rdf_cum.sfu", 'w')
    values = []
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key, val in rdf_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)
    for i in range(len(space)):
        f.write("{:<8.3f} ".format(space[i]/10.)) #10 A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i]))
        f.write("\n")
    f.close()
    
def write_pair(space, pair_dict, props):
    f = open(NAME + "_pair.sfu", 'w')
    values = []
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space(nm) ")
    for key, val in pair_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)
    for i in range(len(space)):
        f.write("{:<8.3f} ".format(space[i]/10.)) #10 A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i]))
        f.write("\n")
    f.close()
    
def pipeline_rdf(U, props, sel):
    r, rdfs, rdfs_cum = rdf(U, props, sel)
    write_rdf(r, rdfs, props)
    write_rdf_cum(r, rdfs_cum, props)
    
def pipeline_pair(U, props, sel):
    r, rdfs = pair_rdf(U, props, sel)
    write_pair(r, rdfs, props)
XTC = "NP22dp-53_PRO1-11_FIX.xtc"
TPR = "NP22dp-53_PRO1.tpr"
NAME = XTC[:-8]

import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis import *
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
U = Universe(TPR, XTC)

props_hb = {
'anchor'    : 'name AU AUS AUL',
'start_ps'  : 0,
'stop_ps'   : 1000000,
'd_a_cutoff'         : 3.5, #A cutoff donor-acceptor distance. gmx hbond=3.5
'd_h_a_angle_cutoff' : 150, #deg cutoff donor-H-aceptor angle. gmx hbond=150
'donors_and_h':{"DOP_N1"     : ["resname DOP and name N1", "resname DOP and name H9 H10 H11"],\
                "DOP_O1"     : ["resname DOP and name O1", "resname DOP and name H4        "],\
                "DOP_O2"     : ["resname DOP and name O2", "resname DOP and name H12       "],\

                "PHE_N1"     : ["resname PHE and name N1", "resname PHE and name H9 H10 H11"],\

                "L22_N1"     : ["resname L22 and name N1", "resname L22 and name H18       "]},


'acceptors'    : {"DOP_O1"     : "resname DOP and name O1      ",\
                  "DOP_O2"     : "resname DOP and name O2      ",\
                  "PHE_O1O2"   : "resname PHE and name O1 O2   ",\
                  "L22_O1"     : "resname L22 and name O1      ",\
                  "L22_O2O3O4" : "resname L22 and name O2 O3 O4"}}

def calc_dha_angle(d, h, a):
    hd = d - h
    ha = a - h
    return np.arccos(np.clip(np.dot(hd, ha)/(np.linalg.norm(hd)*np.linalg.norm(ha)), -1, 1))

def hydrogen_bonds(props):
    g_anchor = U.select_atoms(props['anchor'])
    d_h_a_angle_rad = props['d_h_a_angle_cutoff']/180*np.pi
    hbonds = {}
    for d_key, donor_h in props['donors_and_h'].items():
        for a_key, acceptor in props['acceptors'].items():
            pair = "{}-->{}".format(d_key, a_key)
            print(pair)
            hbonds[pair] = []
            g_donor = U.select_atoms(donor_h[0])
            g_h = U.select_atoms(donor_h[1])
            g_acceptor = U.select_atoms(acceptor)

            for ts in U.trajectory:
                if ts.time >= props['stop_ps']:
                    break
                elif ts.time >= props['start_ps'] and ts.time < props['stop_ps']:
                    dists = [[d.ix, a.ix, np.linalg.norm(d.position - a.position)] for d in g_donor.atoms for a in g_acceptor.atoms]
                    dists = [dist for dist in dists if dist[2] <= props['d_a_cutoff']]
                    angles = [[dist[0], dist[1], h.ix, calc_dha_angle(U.atoms.positions[dist[0]], h.position, U.atoms.positions[dist[1]])] for dist in dists for h in g_h.atoms]
                    bond_ndxs = [[*angle[:3]] for angle in angles if angle[3] >= d_h_a_angle_rad]

                    #anchor_dists has the distance between the COM of the donor/aceptor residues and the COM of gold
                    anchor_dists = [[np.linalg.norm(U.atoms[dix].residue.atoms.center_of_mass()-g_anchor.center_of_mass()), np.linalg.norm(U.atoms[aix].residue.atoms.center_of_mass()-g_anchor.center_of_mass())] for dix, aix, hix in bond_ndxs]

                    hbonds[pair].append([bond_ndxs, anchor_dists])

    times = [frame.time for frame in U.trajectory if frame.time >= props['start_ps'] and frame.time < props['stop_ps']]
    return times, hbonds

def write_hb(props, times, hb):
    f = open(NAME + "_hbonds.sfu", 'w')
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("NDX-donor  NDX-hydrogen  NDX-acceptor InterCOM-distance-anchor-donor InterCOM-distance-anchor-acceptor\n")
    for pair, val in hb.items():
        f.write("#BondPair: {}\n".format(pair))
        for t, (time, bond_data) in enumerate(zip(times, val)):
            f.write("#Time: {} ps\n".format(time))
            for ndx, dist in zip(*bond_data):
                f.write("{:<8d} {:<8d} {:<8d} {:>10.3f} {:>10.3f}\n".format(*ndx, *dist))
    f.close()

times, hbs = hydrogen_bonds(props_hb)
write_hb(props_hb, times, hbs)

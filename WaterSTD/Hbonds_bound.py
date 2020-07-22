"""
Calculates H-bonds only during frames when the donor's COM is close enough to the anchor's (Au) COM
"""

XTC = "NP22sp-53_PRO1-10_FIX.xtc"
TPR = "NP22sp-53_PRO1.tpr"
NAME = XTC[:-8]

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from MDAnalysis import *
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
U = Universe(TPR, XTC)

props_hb_bound = {
'anchor'    : 'name AU AUS AUL',
'd_bound'   : 25, #A, inter-COM (donor-anchor) distance threshold to determine if the donor is bound to the anchor (gold)
'd_a_cutoff'         : 3.5, #A cutoff donor-acceptor distance. gmx hbond=3.5
'd_h_a_angle_cutoff' : 150, #deg cutoff donor-H-aceptor angle. gmx hbond=150

'donors_and_h':{"SER_N1"     : ["resname SER and name N1", "resname SER and name H5 H6 H7  "],\
                "SER_N2"     : ["resname SER and name N2", "resname SER and name H9        "],\
                "SER_O1"     : ["resname SER and name O1", "resname SER and name H13       "],\

                "PHE_N1"     : ["resname PHE and name N1", "resname PHE and name H9 H10 H11"],\

                "L22_N1"     : ["resname L22 and name N1", "resname L22 and name H18       "]},


'acceptors'    : {"SER_O1"     : "resname SER and name O1      ",\
                  "PHE_O1O2"   : "resname PHE and name O1 O2   ",\
                  "L22_O1"     : "resname L22 and name O1      ",\
                  "L22_O2O3O4" : "resname L22 and name O2 O3 O4"}
}

def hbonds_bound(props):
    g_anchor = U.select_atoms(props['anchor'])
    hbonds = defaultdict(list)
    for d_key, donor_h in props['donors_and_h'].items():
        d_residues = U.select_atoms(donor_h[0]).residues
        for a_key, acceptor in props['acceptors'].items():
            pair = "{}-->{}".format(d_key, a_key)
            print(pair)
            for d_res in d_residues[:1]:
                hba = HBA(universe=U,
                          donors_sel=donor_h[0] + " and resid {}".format(d_res.resid),
                          hydrogens_sel=donor_h[1] + " and resid {}".format(d_res.resid),
                          acceptors_sel=acceptor,
                          d_a_cutoff=props['d_a_cutoff'],
                          d_h_a_angle_cutoff=props['d_h_a_angle_cutoff'],
                          update_selections=False)
                hba.run()
                dists_d_anchor = np.array([np.linalg.norm(d_res.atoms.center_of_mass()-g_anchor.center_of_mass()) for ts in U.trajectory])
                bound_mask = dists_d_anchor < props['d_bound']
                try:
                    hba_time = hba.count_by_time()
                    hba_time_masked = list(hba_time[bound_mask])
                except:
                    hba_time_masked = []
                hbonds[pair] += hba_time_masked
            if hbonds[pair] == []:
                hbonds[pair] = (0.0, 0.0)
            else:
                hbonds[pair] = (np.mean(hbonds[pair]), np.std(hbonds[pair]))
    return hbonds

def write_hb_bound(props, hb):
    f = open(NAME + "_hbonds.sfu", 'w')
    f.write("Average (and deviation) of the hydrogen bonds present when the donor is bound to the anchor (donorCOM - anchorCOM < d_bound)\n")
    f.write("The result is averaged over time but it is NOT normalized for the number of donor or number of hydrogen atom. Thus, the means can be greater than 1\n")
    f.write("The numbers are mean hbonds per frame between groups\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Donor-->Acceptor Mean Std\n")
    for key, val in hb.items():
        f.write("{:<30} {:>8.3f} {:>8.3f}\n".format(key,val[0], val[1]))
    f.close()

hbs = hbonds_bound(props_hb_bound)
write_hb_bound(props_hb_bound, hbs)

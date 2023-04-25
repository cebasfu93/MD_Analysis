"""Script for running all analysis on gold and platinum simulations."""
import os
from pathlib import Path

from MDAnalysis import Universe

from aupt.ensemble.binding_times import binding_time
from aupt.ensemble.inputs import (BindingTimesInput,
                                  RadialDistributionFunctionsInput)
from aupt.ensemble.radial_distribution_functions import rdf
from aupt.time_series.contacts import number_of_contacts, number_of_salt_bridges
from aupt.time_series.hydrogen_bonds import hydrogen_bonds
from aupt.time_series.inputs import (ContactsNumberInput,
                                     HydrogenBondsInput,
                                     SaltBridgesNumberInput)
from aupt.utils import get_atoms_with_n_neighbors
from aupt.base_io import BaseWriter

NP_SIZE = 50  # angstrom, as in the directory name
METAL = "AU"
CIT_TYPE = "CIT"
CIT_CONC = 92  # mM
BASE_NAME = f"{METAL.lower()}{NP_SIZE}{CIT_TYPE.lower()}{CIT_CONC:0>3}"
ROOT_DIR = Path(os.getcwd())
MD_PATH = ROOT_DIR

TPR_PATH = MD_PATH / f"{BASE_NAME}_PRO1.tpr"
XTC_PATH = MD_PATH / f"{BASE_NAME}_PRO1-5_FIX.xtc"
UNIVERSE = Universe(TPR_PATH, XTC_PATH)
DT = UNIVERSE.trajectory.dt

# global
CORONA_WIDTH = 15  # A. distance from NP surface considered as the citrate corona
START_TIME = 50_000  # only used for non-time series (e.g., RDF)
STOP_TIME = 200_000

# contacts nalysis
CONTACT_DISTANCE_THRESHOLD = 4.0

# Hbond analysis
DONORS_SEL = f"(same resid as (resname {CIT_TYPE} and (around {CORONA_WIDTH} resname {METAL})))"\
              " and name O1"
HYDROGEN_SEL = f"(same resid as (resname {CIT_TYPE} and (around {CORONA_WIDTH} resname {METAL})))"\
                "and name H5"
ACCEPTORS_SEL = f"(same resid as (resname {CIT_TYPE} and (around {CORONA_WIDTH} resname {METAL})))"\
                 " and name O1 O2 O3 O4 O5 O6 O7"

# salt bridges analysis
CATIONS_SEL = f"(same resid as (resname NA and (around {CORONA_WIDTH} resname {METAL})))"
ANIONS_SEL = f"(same resid as (resname {CIT_TYPE} and (around {CORONA_WIDTH} resname {METAL})))"\
              " and name O2 O3 O4 O5 O6 O7"

# triggers
RUN_RDF = True
RUN_RDF_SURF = True
RUN_BINDING_TIME = True
RUN_CONTACTS_NUMBER = True
RUN_HBONDS = True
RUN_SALT_BRIDGES = True

# atom groups definition
NEIGHBOR_DISTANCE_THRESHOLD = 3.5

# atom groups
ATOM_GROUPS = {
    "NP":  UNIVERSE.select_atoms("resname AU PT"),
    f"{CIT_TYPE}": UNIVERSE.select_atoms(f"resname {CIT_TYPE}"),
    # 1 CIT has 5 Hs. 1 CIH has 6 Hs
    f"{CIT_TYPE}_noh": UNIVERSE.select_atoms(f"resname {CIT_TYPE} and (not name H*)"),
    "NA": UNIVERSE.select_atoms("resname NA"),
    "SOL": UNIVERSE.select_atoms("resname SOL"),
    "SOL_OW": UNIVERSE.select_atoms("resname SOL and name OW"),

    # these selections might vary with CIT_TYPE
    f"{CIT_TYPE}_C1": UNIVERSE.select_atoms(f"resname {CIT_TYPE} and name C1"),
    f"{CIT_TYPE}_C2C3": UNIVERSE.select_atoms(f"resname {CIT_TYPE} and name C2 C3"),
    f"{CIT_TYPE}_C4": UNIVERSE.select_atoms(f"resname {CIT_TYPE} and name C4"),
    f"{CIT_TYPE}_C5C6": UNIVERSE.select_atoms(f"resname {CIT_TYPE} and name C5 C6"),
    f"{CIT_TYPE}_O1": UNIVERSE.select_atoms(f"resname {CIT_TYPE} and name O1"),
    f"{CIT_TYPE}_O2O3": UNIVERSE.select_atoms(f"resname {CIT_TYPE} and name O2 O3"),
    f"{CIT_TYPE}_O4O5O6O7": UNIVERSE.select_atoms(f"resname {CIT_TYPE} and name O4 O5 O6 O7"),
    # salt bridge analysis
    "ANIONS_CORONA": UNIVERSE.select_atoms(ANIONS_SEL, updating=True),
    "CATIONS_CORONA": UNIVERSE.select_atoms(CATIONS_SEL, updating=True)
}
ATOM_GROUPS["FACET_100"] = get_atoms_with_n_neighbors(
    atom_group=ATOM_GROUPS["NP"],
    distance_threshold=NEIGHBOR_DISTANCE_THRESHOLD,
    n_neighbors=8
)
ATOM_GROUPS["FACET_111"] = get_atoms_with_n_neighbors(
    atom_group=ATOM_GROUPS["NP"],
    distance_threshold=NEIGHBOR_DISTANCE_THRESHOLD,
    n_neighbors=9
)
ATOM_GROUPS["NP_EDGES"] = get_atoms_with_n_neighbors(
    atom_group=ATOM_GROUPS["NP"],
    distance_threshold=NEIGHBOR_DISTANCE_THRESHOLD,
    n_neighbors=6
)
ATOM_GROUPS["NP_EDGES"] += get_atoms_with_n_neighbors(
    atom_group=ATOM_GROUPS["NP"],
    distance_threshold=NEIGHBOR_DISTANCE_THRESHOLD,
    n_neighbors=7
)
ATOM_GROUPS["NP_SURF"] = ATOM_GROUPS["FACET_100"] + \
    ATOM_GROUPS["FACET_111"] + ATOM_GROUPS["NP_EDGES"]

ATOM_GROUPS[f"{CIT_TYPE}_NA"] = ATOM_GROUPS[f"{CIT_TYPE}"] + \
    ATOM_GROUPS["NA"]
ATOM_GROUPS[f"{CIT_TYPE}_noh_NA"] = ATOM_GROUPS[f"{CIT_TYPE}_noh"] + \
    ATOM_GROUPS["NA"]


def run_rdf():
    """
    Computes the RDF of atom groups with respect to the NP COM.
    """
    print("COMPUTING RDFS")
    rdf_filename = MD_PATH / f"{XTC_PATH.stem}_rdf.sfu"
    rdf_input = RadialDistributionFunctionsInput(
        r_range=(0, 65),
        nbins=326,
        ref_group_name="NP",
        ref_group_surf=False,
        target_groups_name=["NP", f"{CIT_TYPE}", f"{CIT_TYPE}_noh", "NA",
                            "SOL", "FACET_100", "FACET_111", "NP_EDGES"],
        start_time=START_TIME,
        stop_time=STOP_TIME,
        atom_groups=ATOM_GROUPS
    )
    rdf_output = rdf(
        universe=UNIVERSE,
        input_control=rdf_input
    )
    rdf_writer = BaseWriter(analysis_input=rdf_input,
                            analysis_output=rdf_output)
    rdf_writer.write(
        filename=rdf_filename,
        universe=UNIVERSE
    )

def run_rdf_surf():
    """
    Computes the RDF of atom groups with respect to the surface of the NP, FACET_100, and FACET_111,
    """
    print("COMPUTING RDFS SURF")
    cit_groups = {key: value for key, value in ATOM_GROUPS.items()
                    if f"{CIT_TYPE}_" in key and key != f"{CIT_TYPE}_noh"}
    target_groups_name = list(cit_groups.keys()) + ["NA", "SOL", "SOL_OW"]
    ref_groups_name = ["NP_SURF", "FACET_100", "FACET_111"]
    for ref_group_name in ref_groups_name:
        print(f"REFERENCE: {ref_group_name}")
        rdf_surf_filename = MD_PATH / f"{XTC_PATH.stem}_rdf_{ref_group_name}_surf.sfu"
        rdf_surf_input = RadialDistributionFunctionsInput(
            r_range=(0, 40),
            nbins=201,
            ref_group_name=ref_group_name,
            ref_group_surf=True,
            target_groups_name=target_groups_name,
            start_time=START_TIME,
            stop_time=STOP_TIME,
            atom_groups=ATOM_GROUPS
        )
        rdf_surf_output = rdf(
            universe=UNIVERSE,
            input_control=rdf_surf_input
        )
        rdf_surf_writer = BaseWriter(analysis_input=rdf_surf_input,
                                        analysis_output=rdf_surf_output)
        rdf_surf_writer.write(
            filename=rdf_surf_filename,
            universe=UNIVERSE
    )

def run_binding_times():
    """
    Compute the binding times of given atom groups with NP, FACET_100, and FACET_111.
    """
    print("COMPUTING BINDING TIMES")
    ref_groups_name = ["NP", "FACET_100", "FACET_111"]
    for ref_group_name in ref_groups_name:
        print(f"REFERENCE: {ref_group_name}")
        binding_time_filename = MD_PATH / \
            f"{XTC_PATH.stem}_btimes_{ref_group_name}.sfu"
        binding_times_input = BindingTimesInput(
            ref_group_name=ref_group_name,
            ref_group_com=False,
            target_group_name=f"{CIT_TYPE}_NA",
            target_group_rescom=False,
            distance_threshold=CONTACT_DISTANCE_THRESHOLD,
            start_time=START_TIME,
            stop_time=STOP_TIME,
            atom_groups=ATOM_GROUPS
        )
        binding_time_output = binding_time(
            universe=UNIVERSE,
            input_control=binding_times_input
        )
        binding_time_writer = BaseWriter(analysis_input=binding_times_input,
                                         analysis_output=binding_time_output)
        binding_time_writer.write(
            filename=binding_time_filename,
            universe=UNIVERSE
        )

def run_contacts_number():
    """
    Computes the number of atom-atom and residue-group contacts formed with 
    NP, FACET_100, and FACET_111,
    """
    print("COMPUTING NUMBER OF CONTACTS")
    ref_groups_name = ["NP", "FACET_100", "FACET_111"]
    for ref_group_name in ref_groups_name:
        print(f"REFERENCE: {ref_group_name}")
        contacts_number_filename = MD_PATH / f"{XTC_PATH.stem}_contacts_{ref_group_name}.sfu"
        contacts_number_input = ContactsNumberInput(
            ref_group_name=ref_group_name,
            target_groups_name=[f"{CIT_TYPE}", f"{CIT_TYPE}_noh", "NA"],
            distance_threshold=CONTACT_DISTANCE_THRESHOLD,
            start_time=0,
            stop_time=STOP_TIME,
            atom_groups=ATOM_GROUPS
        )
        contacts_number_output = number_of_contacts(
            universe=UNIVERSE,
            input_control=contacts_number_input
        )
        contacts_number_writer = BaseWriter(analysis_input=contacts_number_input,
                                            analysis_output=contacts_number_output)
        contacts_number_writer.write(
            filename=contacts_number_filename,
            universe=UNIVERSE
        )

def run_hbonds():
    """
    Computes number of hydrogen bonds between groups whose selection is updated every frame.
    """
    print("COMPUTING NUMBER OF HBONDS")
    hbonds_filename = MD_PATH / f"{XTC_PATH.stem}_hbonds.sfu"
    hbonds_input = HydrogenBondsInput(
        donors_sel=DONORS_SEL,
        hydrogens_sel=HYDROGEN_SEL,
        acceptors_sel=ACCEPTORS_SEL,
        d_a_cutoff=3.0,
        d_h_cutoff=1.2,
        d_h_a_angle_cutoff=150,
        update_selections=True,
        start_time=0,
        stop_time=STOP_TIME
    )
    hbonds_output = hydrogen_bonds(
        universe=UNIVERSE,
        input_control=hbonds_input
    )
    hbonds_writer = BaseWriter(analysis_input=hbonds_input,
                               analysis_output=hbonds_output)
    hbonds_writer.write(
        filename=hbonds_filename,
        universe=UNIVERSE
    )

def run_salt_bridges():
    """
    Computes number of salt bridges between updating atom groups.
    """
    print("COMPUTING SALT BRIDGES")
    bridges_filename  = MD_PATH / f"{XTC_PATH.stem}_bridges.sfu"
    bridges_input = SaltBridgesNumberInput(
        anions_group_name="ANIONS_CORONA",
        cations_group_name="CATIONS_CORONA",
        distance_threshold=3.0,
        start_time=0,
        stop_time=STOP_TIME,
        atom_groups=ATOM_GROUPS
    )
    bridges_output = number_of_salt_bridges(
        universe=UNIVERSE,
        input_control=bridges_input
    )
    bridges_writer = BaseWriter(analysis_input=bridges_input,
                                analysis_output=bridges_output)
    bridges_writer.write(
        filename=bridges_filename,
        universe=UNIVERSE
    )

if __name__ == "__main__":
    if RUN_RDF is True:
        run_rdf()

    if RUN_RDF_SURF is True:
        run_rdf_surf()

    if RUN_BINDING_TIME is True:
        run_binding_times()

    if RUN_CONTACTS_NUMBER is True:
        run_contacts_number()

    if RUN_HBONDS is True:
        run_hbonds()

    if RUN_SALT_BRIDGES is True:
        run_salt_bridges()

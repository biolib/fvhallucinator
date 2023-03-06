import os
import torch
import pyrosetta
import time
from .mds import build_fv_mds
from .score_functions import *
from .utils import get_constraint_set_mover, resolve_clashes, disulfidize
from src.deepab.constraints import ConstraintType, get_constraint_residue_pairs, get_filtered_constraint_file
from src.deepab.constraints.custom_filters import hb_dist_filter
from src.deepab.constraints.rosetta_constraint_generators import logit_to_energy
from src.deepab.models.AbResNet import AbResNet
from src.util.model_out import get_probs_from_model, bin_matrix, binned_mat_to_values
from src.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins
from src.util.util import get_heavy_seq_len
pyrosetta.init('--mute all')

def build_initial_fv(fasta_file: str,
                     mds_pdb_file: str,
                     model: AbResNet,
                     mask_distant_orientations: bool = True) -> None:
    """
    Build initial Fv structure from AbResNet outputs via MDS
    """

    masked_bin_num = 1 if mask_distant_orientations else 0
    bins = [
        get_dist_bins(model._num_out_bins),
        get_dist_bins(model._num_out_bins),
        get_dist_bins(model._num_out_bins),
        get_dihedral_bins(model._num_out_bins - masked_bin_num, rad=True),
        get_dihedral_bins(model._num_out_bins - masked_bin_num, rad=True),
        get_planar_bins(model._num_out_bins - masked_bin_num, rad=True)
    ]

    getting_probs_from_model_time = time.time()
    print("Getting probs from model")
    probs = get_probs_from_model(model, fasta_file)
    print(f'Got probs from models in {time.time() - getting_probs_from_model_time}s')
    pred_bin_mats = [bin_matrix(p, are_logits=False) for p in probs]
    pred_value_mats = [
        binned_mat_to_values(pred_bin_mats[i], bins[i])
        for i in range(len(pred_bin_mats))
    ]

    dist_mask = pred_value_mats[1] < bins[1][-1][0]

    print('Building FV Mds')
    mds_time = time.time()
    build_fv_mds(fasta_file,
                 mds_pdb_file,
                 pred_value_mats[1],
                 pred_value_mats[3],
                 pred_value_mats[4],
                 pred_value_mats[5],
                 mask=dist_mask)
    print(f'MDS built in {time.time() - mds_time}')

    pose = pyrosetta.pose_from_pdb(mds_pdb_file)
    disulfidize(pose, pred_value_mats[1])
    pose.dump_pdb(mds_pdb_file)


def get_cst_file(model: torch.nn.Module, fasta_file: str,
                 constraint_dir: str) -> str:
    """
    Generate standard constraint files for Fv builder
    """

    heavy_seq_len = get_heavy_seq_len(fasta_file)
    print("Getting residue pairs")
    res_time = time.time()
    residue_pairs = get_constraint_residue_pairs(model,
                                                 fasta_file,
                                                 heavy_seq_len,
                                                 use_logits=True)

    print(f"Got residue pairs in {time.time() - res_time}")

    print("Getting cst file")
    all_cst_file = get_filtered_constraint_file(
        residue_pairs=residue_pairs,
        constraint_dir=os.path.join(constraint_dir, "all_csm"),
        threshold=0.1,
        constraint_types=[
            ConstraintType.cb_distance, ConstraintType.ca_distance,
            ConstraintType.omega_dihedral, ConstraintType.theta_dihedral,
            ConstraintType.phi_planar
        ],
        prob_to_energy=logit_to_energy)

    print("Getting HB file")
    hb_cst_file = get_filtered_constraint_file(
        residue_pairs=residue_pairs,
        constraint_dir=os.path.join(constraint_dir, "hb_csm"),
        local=True,
        threshold=0.3,
        constraint_types=[ConstraintType.no_distance],
        constraint_filters=[hb_dist_filter],
        prob_to_energy=logit_to_energy)

    os.system("cat {} >> {}".format(all_cst_file, hb_cst_file))

    return hb_cst_file


def get_centroid_min_mover(
        max_iter: int = 1000,
        repeats: int = 6) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create centroid minimization mover
    """

    sf = get_sf_cen()
    sf_cart = get_sf_cart()

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(max_iter)

    min_mover_cart = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf_cart, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover_cart.max_iter(max_iter)
    min_mover_cart.cartesian(True)

    repeat_mover = pyrosetta.rosetta.protocols.moves.RepeatMover(
        min_mover, repeats)

    centroid_fold_mover = pyrosetta.rosetta.protocols.moves.SequenceMover()
    centroid_fold_mover.add_mover(repeat_mover)
    centroid_fold_mover.add_mover(min_mover_cart)

    return centroid_fold_mover


def get_fa_relax_mover(
        max_iter: int = 200) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom relax mover
    """

    sf_fa = get_sf_fa()

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(max_iter)
    relax.dualspace(True)
    relax.set_movemap(mmap)
    relax.ramp_down_constraints(True)

    return relax


def get_fa_min_mover(
        max_iter: int = 1000) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom minimization mover
    """

    sf_fa = get_sf_fa()
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 1)
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0)

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf_fa, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(max_iter)
    min_mover.cartesian(True)

    return min_mover


def refine_fv(in_pdb_file: str,
              out_pdb_file: str,
              cst_file: str,
              verbose: bool = True) -> float:
    """
    Run constrained minimization protocol on initial pdb file and return final score
    """
    ############################################################################
    # Load initial pose
    ############################################################################
    print("Reading pdb for Fv Refinement..")
    time_read = time.time()
    pose = pyrosetta.pose_from_pdb(in_pdb_file)

    switch_cen = pyrosetta.rosetta.protocols.simple_moves.SwitchResidueTypeSetMover(
        "centroid")
    switch_fa = pyrosetta.rosetta.protocols.simple_moves.SwitchResidueTypeSetMover(
        "fa_standard")
    switch_cen.apply(pose)

    csm = get_constraint_set_mover(cst_file)
    csm.apply(pose)
    print(f"Read PDB and applied constraints in: {time.time() - time_read}")

    ############################################################################
    # Apply global centroid movers
    ############################################################################
    if verbose:
        print('centroid minimize...')

    centroid_time = time.time()
    centroid_fold_mover = get_centroid_min_mover()
    switch_cen.apply(pose)
    centroid_fold_mover.apply(pose)
    print(f"Centroid minimization done in {time.time() - centroid_time}")
    print("Removing clashes from pose...")
    clash_time = time.time()
    resolve_clashes(pose)
    print(f"Removed clashes in {time.time() - clash_time}")


    ############################################################################
    # Apply full atom relax
    ############################################################################
    if verbose:
        print('full atom relax...')

    relax_time = time.time()
    fa_relax_mover = get_fa_relax_mover()
    switch_fa.apply(pose)
    fa_relax_mover.apply(pose)
    print(f"Did full atom relaxation in {time.time() - relax_time}")

    ############################################################################
    # Apply full atom refinement
    ############################################################################
    if verbose:
        print("full atom minimize...")

    fa_min_time = time.time()
    fa_min_mover = get_fa_min_mover()
    fa_min_mover.apply(pose)

    pose.dump_pdb(out_pdb_file)

    sf_fa_cst = get_sf_fa()
    score = sf_fa_cst(pose)

    print(f"Did full atom minimization in {time.time() - fa_min_time}")
    
    if verbose:
       print("score ", score)

    return score

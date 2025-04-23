import os
import autode as ade
import torch
import pickle
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np

from qm9 import bond_analyze
from rdkit import Chem
from qm9.rdkit_functions import BasicMolecularMetrics

use_rdkit = True


####################################################################
###################### Helper Functions ############################
####################################################################

def _geoldm_metrics(mol_list, dataset_info, metrics, kwd):
    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    for mol in mol_list:
        pos, atom_type = mol
        validity_results = _check_stability(pos, atom_type, dataset_info)
        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])
    # fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    rdkit_metrics_list = metrics.evaluate(mol_list)
    geoldm_metric_dict = {
        f'{kwd}_geoldm_mol_stable': molecule_stable,
        f'{kwd}_geoldm_atm_stable': nr_stable_bonds,
        f'{kwd}_geoldm_total_atoms': n_atoms,
        f'{kwd}_geoldm_valid': rdkit_metrics_list[1][0],
        f'{kwd}_geoldm_unique': rdkit_metrics_list[1][1],
        f'{kwd}_geoldm_novel': rdkit_metrics_list[1][2]
    }
    return geoldm_metric_dict


def _check_stability(positions, atom_type, dataset_info, debug=False):
    """
    Adapt from GeoLDM
    """
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            # str type of atom
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or \
                    dataset_info['name'] == 'qm9_first_half':
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                order = bond_analyze.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)

    return molecule_stable, nr_stable_bonds, len(x)


def _parse_file(filename):
    """
    Read from file and return an ade `Molecule` object
    """
    atom_list = []
    with open(filename, "r") as file:
        for line in file.readlines():
            content = line.split(" ")
            if len(content) != 4:
                continue
            atom_list.append(ade.Atom(content[0], x=content[1], y=content[2], z=content[3]))
    return ade.Molecule(atoms=atom_list)


def _txt_mol_to_tensor(filename, dataset_info):
    """
    Read from file and return a list of tensors as [..., (position, feature), ...]
    """
    atom_type_list = []
    atom_coord_list = []
    with open(filename, "r") as file:
        for line in file.readlines():
            content = line.split()
            if len(content) != 4:
                continue
            atom_type_list.append(dataset_info['atom_decoder'].index(content[0]))
            atom_coord_list.append([float(content[1]), float(content[2]), float(content[3])])
    return [torch.tensor(atom_coord_list), torch.tensor(atom_type_list)]


def _calculate_metrics(molecule, kwd: str, empty=False):
    """

    Parameters
    ----------
    molecule: ade.Molecule object
    kwd: before or after

    Returns
    -------
    Dictionary of metrics
    """
    if empty:
        return {f"{kwd}_l1_force": None, f"{kwd}_rms_force": None, f"{kwd}_energy": None,
                f"{kwd}_weighted_l1_force": None, f"{kwd}_weighted_rms_force": None,
                f"{kwd}_weighted_energy": None, f"{kwd}_molecule_weight": None, f"{kwd}_valid": False}
    gradient = torch.tensor(molecule.gradient.tolist())
    l1_force = gradient.abs().sum().item()
    rms_force = torch.sqrt(torch.square(gradient).sum() / len(gradient)).item()
    weights = torch.tensor([float(mass) for mass in molecule.atomic_masses]).view(-1, 1)
    total_weight = weights.sum().item()
    weights /= total_weight
    weighted_gradient = gradient * weights
    weighted_l1_force = weighted_gradient.abs().sum().item()
    weighted_rms_force = torch.sqrt(torch.square(weighted_gradient).sum() / len(weighted_gradient)).item()
    energy = float(molecule.energy)
    metric_dict = {f"{kwd}_l1_force": l1_force, f"{kwd}_rms_force": rms_force, f"{kwd}_energy": energy,
                   f"{kwd}_weighted_l1_force": weighted_l1_force, f"{kwd}_weighted_rms_force": weighted_rms_force,
                   f"{kwd}_weighted_energy": energy * total_weight, f"{kwd}_molecule_weight": total_weight,
                   f"{kwd}_valid": True}
    return metric_dict


####################################################################
############# use this function for calculate metric ###############
####################################################################

def calculate_metric(file_path: str, dataset_info: dict, metric: BasicMolecularMetrics) -> dict:
    """
    Given some molecules, calculate the metrics as reported in Table 1

    Parameters
    ----------
    file_path: path to a folder of molecule files
    dataset_info: dict in configs/datasets_config.py, see example in main function
    metric: qm9.rdkit_functions.BasicMolecularMetrics object, see example in main function

    Returns
    -------
    a dict of all metrics
    """
    if os.path.exists(f"final_saved.pickle"):
        print(f"Use pre-existing .pickle file {file_path}")
        with open(f"final_saved.pickle", 'rb') as handle:
            save_dict = pickle.load(handle)
        return save_dict
    all_files = os.listdir(os.getcwd())
    all_files = list(filter(lambda x: "molecule" in x or "conditional" in x, all_files))
    all_files.sort()
    before_metric_list = []
    after_metric_list = []
    before_mol_list = []
    after_mol_list = []

    before_total_valid_file = 0
    after_total_valid_file = 0
    optimization_success_num = 0
    temp_dir = f"xyz_temp"
    here = os.getcwd()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    os.chdir(temp_dir)
    print(os.getcwd())
    # run some molecules and get their energy and force
    for i, mol_file in tqdm(enumerate(all_files)):
        before_valid_flag = False
        after_valid_flag = False

        try:
            molecule = _parse_file(os.path.join("..", mol_file))
            before_mol_list.append(_txt_mol_to_tensor(os.path.join("..", mol_file), dataset_info))
            input_filename = f"temp.xyz"
            optimized_input_filename = f"optimized_temp.xyz"
            if os.path.exists("gradient"):
                os.remove("gradient")
            if os.path.exists("energy"):
                os.remove("energy")
            if os.path.exists(input_filename):
                os.remove(input_filename)
            molecule.print_xyz_file(filename=input_filename, append=False)
            molecule.single_point(method=ade.methods.XTB(), keywords=["--grad"],
                                  n_cores=4, temp_dir=temp_dir, input_filename=input_filename)
            # force = torch.tensor(molecule.gradient.tolist()).abs().sum()
            before_metric_dict = _calculate_metrics(molecule, kwd="before")
            before_metric_list.append(before_metric_dict)
            before_total_valid_file += 1
            before_valid_flag = True

            if os.path.exists(optimized_input_filename):
                os.remove(optimized_input_filename)
            # just to be consistent, add a valance check after optimization
            molecule.optimise(method=ade.methods.XTB(), temp_dir=temp_dir, input_filename=input_filename,
                              reset_graph=True, optimized_filename=optimized_input_filename)
            optimization_success_num += 1
            after_mol_list.append(_txt_mol_to_tensor(optimized_input_filename, dataset_info))
            if os.path.exists("gradient"):
                os.remove("gradient")
            if os.path.exists("energy"):
                os.remove("energy")
            if os.path.exists(input_filename):
                os.remove(input_filename)
            molecule.single_point(method=ade.methods.XTB(), keywords=["--grad"],
                                  n_cores=4, temp_dir=temp_dir, input_filename=optimized_input_filename)
            after_metric_dict = _calculate_metrics(molecule, kwd="after")
            after_metric_list.append(after_metric_dict)
            after_total_valid_file += 1
            after_valid_flag = True
        except Exception as e:
            pass
            print(f"[{i}] {e}")
        finally:
            # something wrong with after, can't optimize or can't calculate gradient
            if before_total_valid_file > after_total_valid_file:
                # can't calculate gradient after optimization
                if len(after_mol_list) == len(before_mol_list):
                    exit("XTB error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # can't optimize
                elif len(after_mol_list) > len(before_mol_list):
                    print(f"[{i}] OPTIMIZATION FAILED   OPTIMIZATION FAILED   OPTIMIZATION FAILED")
                    # after_metric_list.append(calculate_metrics(None, kwd="after", empty=True))
            elif before_total_valid_file < after_total_valid_file:
                exit("something is wrong !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # break
            if not before_valid_flag:
                before_metric_list.append(_calculate_metrics(None, kwd="before", empty=True))
            if not after_valid_flag:
                after_metric_list.append(_calculate_metrics(None, kwd="after", empty=True))

    os.chdir(here)
    shutil.rmtree(temp_dir)
    final_before_df = pd.DataFrame(before_metric_list)
    final_after_df = pd.DataFrame(after_metric_list)
    final_before_df.to_csv("raw_before_data.csv", index=False)
    final_after_df.to_csv("raw_after_data.csv", index=False)

    final_dict = {}
    for col in final_before_df.columns:
        if sum(final_before_df["before_valid"]) != 0:
            final_dict[col] = sum(final_before_df[col].dropna()) / sum(final_before_df["before_valid"])
        else:
            final_dict[col] = 0.
    for col in final_after_df.columns:
        if sum(final_after_df["after_valid"]) != 0:
            final_dict[col] = sum(final_after_df[col].dropna()) / sum(final_after_df["after_valid"])
        else:
            final_dict[col] = 0.

    final_dict["before_weighted_energy"] = final_dict["before_weighted_energy"] / final_dict[
        "before_molecule_weight"] if final_dict["before_molecule_weight"] != 0 else 0.
    final_dict["after_weighted_energy"] = final_dict["after_weighted_energy"] / final_dict["after_molecule_weight"] if \
        final_dict["after_molecule_weight"] != 0 else 0.
    final_dict["before_valid"] = before_total_valid_file
    final_dict["after_valid"] = after_total_valid_file
    final_dict["delta_valid"] = after_total_valid_file - before_total_valid_file

    delta_valid_idx = np.array(final_after_df["after_valid"])
    final_dict["optimization_success"] = optimization_success_num
    for col in ["energy", "weighted_energy", "l1_force", "weighted_l1_force", "rms_force", "weighted_rms_force"]:
        delta_list = np.array(final_after_df[f"after_{col}"])[delta_valid_idx] - \
                     np.array(final_before_df[f"before_{col}"])[delta_valid_idx]
        if len(delta_list) == 0:
            final_dict[f"delta_{col}"] = 0
        else:
            final_dict[f"delta_{col}"] = delta_list.mean().item()

    final_dict["before_total_molecules"] = len(before_mol_list)
    final_dict["after_total_molecules"] = len(after_mol_list)
    final_dict["delta_total_molecules"] = len(after_mol_list) - len(before_mol_list)

    geoldm_before_metric = _geoldm_metrics(before_mol_list, dataset_info, metric, kwd="before")
    geoldm_after_metric = _geoldm_metrics(after_mol_list, dataset_info, metric, kwd="after")
    final_dict.update(geoldm_before_metric)
    final_dict.update(geoldm_after_metric)

    final_dict["delta_geoldm_mol_stable"] = final_dict["after_geoldm_mol_stable"] - final_dict[
        "before_geoldm_mol_stable"]
    final_dict["delta_geoldm_atm_stable"] = final_dict["after_geoldm_atm_stable"] - final_dict[
        "before_geoldm_atm_stable"]
    final_dict["delta_geoldm_total_atoms"] = final_dict["after_geoldm_total_atoms"] - final_dict[
        "before_geoldm_total_atoms"]
    final_dict["delta_geoldm_valid"] = final_dict["after_geoldm_valid"] - final_dict["before_geoldm_valid"]
    final_dict["delta_geoldm_unique"] = final_dict["after_geoldm_unique"] - final_dict["before_geoldm_unique"]
    final_dict["delta_geoldm_novel"] = final_dict["after_geoldm_novel"] - final_dict["before_geoldm_novel"]

    with open(f"final_saved.pickle", "wb") as handle:
        pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return final_dict


if __name__ == "__main__":
    # here is an example of how to calculate metrics
    # you can use molecules in "molecule_examples" to play around
    from configs.datasets_config import get_dataset_info

    # Step 1: specify dataset name and the directory to the generated molecules
    eval_folder_dir = "molecule_examples"  # please change this to your output directory
    dataset_name = "qm9"  # or "geom"
    rdkit_metrics_dir = f"some/saving/directory/{dataset_name}_rdkit_metrics.pickle"  # please specify some/saving/directory

    # Step 2: manual preparation work
    if dataset_name == "geom":
        with open(os.path.join("outputs/drugs_latent2", 'args.pickle'), 'rb') as f:
            args = pickle.load(f)
    else:
        with open(os.path.join("outputs/qm9_latent2", 'args.pickle'), 'rb') as f:
            args = pickle.load(f)
    input_dataset_info = get_dataset_info(args.dataset, args.remove_h)
    if os.path.exists(rdkit_metrics_dir):
        with open(rdkit_metrics_dir, 'rb') as handle:
            rdkit_metrics = pickle.load(handle)
    else:
        rdkit_metrics = BasicMolecularMetrics(input_dataset_info)
        with open(rdkit_metrics_dir, "wb") as handle:
            pickle.dump(rdkit_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Step 3: just this one line to calculate all metrics
    result_dict = calculate_metric(eval_folder_dir, input_dataset_info, rdkit_metrics)
    print(result_dict)

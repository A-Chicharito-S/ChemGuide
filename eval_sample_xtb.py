# Rdkit import should be first, do not move it
import time

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import os
import utils
import random
import numpy as np
import argparse
from qm9 import dataset
from qm9.models_xtb import get_latent_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
import torch
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from qm9.sampling_xtb import sample_chain, sample
from configs.datasets_config import get_dataset_info


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def save_and_sample_chain(args, eval_args, device, flow,
                          n_tries, n_nodes, dataset_info, id_from=0,
                          num_chains=100):
    for i in range(num_chains):
        target_path = f'eval/chain_{i}/'

        one_hot, charges, x = sample_chain(
            args, device, flow, n_tries, dataset_info)

        vis.save_xyz_file(
            join(eval_args.model_path, target_path), one_hot, charges, x,
            dataset_info, id_from, name='chain')

        vis.visualize_chain_uncertainty(
            join(eval_args.model_path, target_path), dataset_info,
            spheres_3d=True)

    return one_hot, charges, x


def sample_different_sizes_and_save(args, eval_args, device, generative_model,
                                    nodes_dist, dataset_info, target_context=None, nodesxsample=None):
    if nodesxsample is None:
        nodesxsample = nodes_dist.sample(args.num_samples)

    one_hot, charges, x, node_mask = sample(
        args, device, generative_model, dataset_info,
        nodesxsample=nodesxsample,
        target_context=target_context)

    stable_cnt = 0
    for i, single_node_mask in enumerate(node_mask):
        num_atoms = int(single_node_mask.sum().item())
        atom_type = one_hot[i:i + 1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
        x_squeeze = x[i:i + 1, :num_atoms].squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]
        if mol_stable:
            stable_cnt += 1
    print('num stable molecules {}/{}'.format(stable_cnt, node_mask.shape[0]))

    save_path = f'{generative_model.save_path}/generated_mols/{args.run_id}'
    vis.save_xyz_file(
        save_path, one_hot, charges, x,
        id_from=0, name='molecule', dataset_info=dataset_info,
        node_mask=node_mask)


def sample_only_stable_different_sizes_and_save(args, eval_args, device, flow, nodes_dist, dataset_info):
    assert args.num_tries > args.num_stable_samples

    nodesxsample = nodes_dist.sample(args.num_tries)
    one_hot, charges, x, node_mask = sample(args, device, flow, dataset_info, nodesxsample=nodesxsample)

    counter = 0
    for i in range(args.num_tries):
        num_atoms = int(node_mask[i:i + 1].sum().item())
        atom_type = one_hot[i:i + 1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
        x_squeeze = x[i:i + 1, :num_atoms].squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

        num_remaining_attempts = args.num_tries - i - 1
        num_remaining_samples = args.num_stable_samples - counter

        if mol_stable or num_remaining_attempts <= num_remaining_samples:
            if mol_stable:
                print('Found stable mol.')
            vis.save_xyz_file(
                join(eval_args.model_path, 'eval/molecules/'),
                one_hot[i:i + 1], charges[i:i + 1], x[i:i + 1],
                id_from=counter, name='molecule_stable',
                dataset_info=dataset_info,
                node_mask=node_mask[i:i + 1])
            counter += 1

            if counter >= args.num_stable_samples:
                break


def get_target_context(generators_path, device, args):
    from eval_conditional_qm9 import get_args_gen, compute_mean_mad, get_dataloader

    if 'prop' in args.package and not args.use_cond_geo and not args.use_neural_guidance:  # using xtb for prop
        from qm9.models import DistributionNodes, DistributionProperty
        if 'qm9' in args.model_path:
            pickle_file = 'qm9_xtb_property.pickle'
            print('prop info for qm9 using xtb for property guidance is loaded')
        else:
            pickle_file = 'geom_xtb_property.pickle'
            print('prop info for geom using xtb for property guidance is loaded')
        with open(pickle_file, 'rb') as handle:
            n_atoms_prop_info_xtb_dict = pickle.load(handle)
        n_atoms = n_atoms_prop_info_xtb_dict['n_atoms']
        property_dict = n_atoms_prop_info_xtb_dict['property_dict']

        histogram = {}
        for size in n_atoms:
            if size in histogram.keys():
                histogram[size] += 1
            else:
                histogram[size] = 1
        property_nodes_dist = DistributionNodes(histogram)
        property_norms = None
        property_prop_dist = DistributionProperty(dataloader=None, properties=[args.property], normalizer=None,
                                                  data_num_atoms=torch.tensor(n_atoms),
                                                  data_prop={prop: torch.tensor(val) for prop, val in
                                                             property_dict.items()})
    else:
        args_gen = get_args_gen(generators_path)

        # Careful with this -->
        if not hasattr(args_gen, 'diffusion_noise_precision'):
            args_gen.normalization_factor = 1e-4
        if not hasattr(args_gen, 'normalization_factor'):
            args_gen.normalization_factor = 1
        if not hasattr(args_gen, 'aggregation_method'):
            args_gen.aggregation_method = 'sum'
        property_dataloaders = get_dataloader(args_gen)
        property_norms = compute_mean_mad(property_dataloaders, args_gen.conditioning, args_gen.dataset)
        from eval_conditional_qm9 import get_generator
        gen_model, property_nodes_dist, property_prop_dist, _ = get_generator(generators_path, property_dataloaders,
                                                                              args.device, args_gen, property_norms)
    property_nodesxsample = property_nodes_dist.sample(args.num_samples)

    target_context = property_prop_dist.sample_batch(property_nodesxsample).to(device).view(-1, )

    if property_norms is not None:  # in the case xtb for property is not used
        args.mean = property_norms[args.property]['mean']
        args.mad = property_norms[args.property]['mad']

    return target_context, property_nodesxsample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="outputs/qm9_latent2",
                        help='Specify model path')
    parser.add_argument(
        '--n_tries', type=int, default=10,
        help='N tries to find stable molecule for gif animation')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of atoms in molecule for gif animation')
    parser.add_argument("--clf_scale", type=float, default=1.)
    parser.add_argument("--acc", type=str, default="1.0")
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--n_cpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", type=str, default='run0')
    parser.add_argument("--max_core", type=int, default=2000)
    parser.add_argument("--grad_clip_threshold", type=float, default=1.)  # float("inf")
    parser.add_argument("--guidance_steps", type=int, default=800)
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--num_stable_samples", type=int, default=10)
    parser.add_argument("--num_tries", type=int, default=2 * 10)
    parser.add_argument("--save_traj", type=int, default=0)  # only for debug
    parser.add_argument("--package", type=str, default="xtb", help="xtb or autode")
    parser.add_argument("--every_n_step", type=int, default=2, help="add guidance every n step")
    parser.add_argument("--use_hist_grad", action='store_true',
                        help="whether use the stored grad for every n-1 step, "
                             "if True, among every n step, the grad at step 1 will be used for the rest n-1 steps,"
                             "if False, the grad for rest n-1 step will be 0")
    parser.add_argument("--use_neural_guidance", action='store_true',
                        help="whether use the neural property classifier/regressor")
    parser.add_argument("--cond_neural_model", default=None,
                        help="the conditional function to provide guidance")
    # (neural) guidance property
    parser.add_argument("--property", type=str, default=None,
                        help="a string (e.g., \'mu\') that suggests the property to be guided by the regressor")
    parser.add_argument("--recurrent_times", type=int, default=1,
                        help="the number of times that we do recurrent sampling")
    parser.add_argument("--use_cond_geo", action='store_true',
                        help="whether use conditional GeoLDM model")

    parser.add_argument("--bilevel_opti", action='store_true',
                        help="whether we will use the ")
    parser.add_argument("--bilevel_method", type=str, default='uni_guide',
                        help="a string (e.g., \'uni_guide\') that suggests the bilevel method to be used")
    parser.add_argument("--clf_scale_force", type=float, default=None)
    parser.add_argument("--clf_scale_prop", type=float, default=None)
    # evolutionary algorithm
    parser.add_argument("--use_evolution", action='store_true',
                        help="whether we will use the evolutionary algorithm")
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--check_variants_interval", type=int, default=2)

    eval_args, unparsed_args = parser.parse_known_args()
    seed = eval_args.seed
    if seed is not None:
        print(f'setting seed as {seed}')
        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    assert eval_args.model_path is not None

    if eval_args.use_evolution:
        assert eval_args.num_beams is not None and eval_args.num_beams >= 1
        assert eval_args.check_variants_interval is not None

    if eval_args.n_cpus > os.cpu_count():
        print(f"System only has {os.cpu_count()} but requires {eval_args.n_cpus}")
        eval_args.n_cpus = os.cpu_count()
    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # specify cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('availability of cuda: {}'.format(torch.cuda.is_available()))
    device = torch.device("cuda" if args.cuda else "cpu")

    assert isinstance(eval_args.use_hist_grad, bool)

    # deal with whether to use guidance from neural networks (e.g., a classifier / regressor)
    if not eval_args.use_neural_guidance and not eval_args.use_cond_geo and not eval_args.bilevel_opti:
        # do not use guidance from neural network
        assert eval_args.cond_neural_model is None
    else:
        from qm9.property_prediction.models_property import EGNN
        eval_args.cond_neural_model = EGNN(in_node_nf=5, in_edge_nf=0,
                                           hidden_nf=128,
                                           n_layers=7,
                                           coords_weight=1.0,
                                           attention=1,
                                           node_attr=0,
                                           )  # we will map its device again in sampling_xtb.py

        assert eval_args.property is not None, 'please specify a property you wish to guide'

        supported_properties = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv',]
        if eval_args.property in supported_properties:
            regressor_state_dict = torch.load(
                f='regressors/exp_class_{}/best_checkpoint.npy'.format(eval_args.property), map_location=device)
            eval_args.cond_neural_model.load_state_dict(regressor_state_dict)
            print('==============EGNN for {} loaded!=============='.format(eval_args.property))
        else:
            raise NotImplementedError

    assert eval_args.package is not None or eval_args.property is not None, \
        'please specify at least chemistry guidance (xtb or autode) or neural guidance (property regressor)'

    if eval_args.bilevel_opti:
        assert eval_args.package is not None, 'for bi-level optimization, please specify the chemistry package'
        assert eval_args.property is not None, 'for bi-level optimization, please specify the property to be guided'
        assert eval_args.clf_scale_force is not None, 'for bi-level optimization, please specify the guidance scale for force'
        assert eval_args.clf_scale_prop is not None, 'for bi-level optimization, please specify the guidance scale for force'
    else:
        assert eval_args.clf_scale_force is None, 'reserved only for bi-level optimization'
        assert eval_args.clf_scale_prop is None, 'reserved only for bi-level optimization'

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.model_path = eval_args.model_path
    args.num_samples = eval_args.num_samples
    args.clf_scale = eval_args.clf_scale
    args.guidance_steps = eval_args.guidance_steps
    args.grad_clip_threshold = eval_args.grad_clip_threshold
    args.num_stable_samples = eval_args.num_stable_samples
    args.num_tries = eval_args.num_tries
    args.save_traj = eval_args.save_traj
    args.n_cores = eval_args.n_cores
    args.max_core = eval_args.max_core
    args.acc = eval_args.acc
    args.package = eval_args.package
    args.every_n_step = eval_args.every_n_step
    args.use_hist_grad = eval_args.use_hist_grad
    args.n_cpus = eval_args.n_cpus
    args.seed = eval_args.seed

    args.use_neural_guidance = eval_args.use_neural_guidance
    args.cond_neural_model = eval_args.cond_neural_model
    args.property = eval_args.property
    args.run_id = eval_args.run_id
    args.recurrent_times = eval_args.recurrent_times
    args.bilevel_opti = eval_args.bilevel_opti
    args.bilevel_method = eval_args.bilevel_method
    args.clf_scale_force = eval_args.clf_scale_force
    args.clf_scale_prop = eval_args.clf_scale_prop
    args.use_cond_geo = eval_args.use_cond_geo
    # evolutionary algorithm
    args.num_beams = eval_args.num_beams
    args.check_variants_interval = eval_args.check_variants_interval
    args.use_evolution = eval_args.use_evolution
    args.mean = None
    args.mad = None

    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    flow, nodes_dist, prop_dist = get_latent_diffusion(
        args, device, dataset_info, dataloaders['train'])
    flow.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn),
                                 map_location=device)

    flow.load_state_dict(flow_state_dict)

    print('Sampling handful of molecules.')

    if args.use_cond_geo:
        assert args.property is not None
        assert args.property in eval_args.model_path
    print(
        f'property={args.property}, use cond-geoldm={args.use_cond_geo}, use neural guidance={args.use_neural_guidance}')
    if args.property is not None or args.use_cond_geo:
        generators_path = "outputs/exp_cond_{}".format(args.property)
        target_context, nodesxsample = get_target_context(generators_path, device, args)
        if 'prop' not in args.package:  # if using xtb for property, then mean & mad should be None
            assert args.mean is not None
            assert args.mad is not None
    else:
        target_context, nodesxsample = None, None

    sample_different_sizes_and_save(
        args, eval_args, device, flow, nodes_dist,
        dataset_info=dataset_info,
        target_context=target_context,
        nodesxsample=nodesxsample)
    exit()
    #
    # print('Sampling stable molecules.')
    # sample_only_stable_different_sizes_and_save(
    #     args, eval_args, device, flow, nodes_dist,
    #     dataset_info=dataset_info)
    # print('Visualizing molecules.')
    # vis.visualize(
    #     join(eval_args.model_path, 'eval/molecules/'), dataset_info,
    #     max_num=100, spheres_3d=True)
    #
    # print('Sampling visualization chain.')
    # save_and_sample_chain(
    #     args, eval_args, device, flow,
    #     n_tries=eval_args.n_tries, n_nodes=eval_args.n_nodes,
    #     dataset_info=dataset_info)


if __name__ == "__main__":
    main()

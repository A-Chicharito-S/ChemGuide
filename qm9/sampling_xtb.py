import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
    assert_correctly_masked
from qm9.analyze import check_stability


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None):
    n_samples = 1
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    else:
        raise ValueError()

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
        context = context.repeat(1, n_nodes, 1).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False, target_context=None):

    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    atom2index = dataset_info['atom_encoder']  # e.g., {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    atom_syb2num = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11,
                    "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "Ca": 20, "Sc": 21, "Ti": 22,
                    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
                    "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
                    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
                    "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62,
                    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72,
                    "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
                    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
                    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101,
                    "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
                    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
                    }
    atom_mapping = {val: atom_syb2num[key] for key, val in atom2index.items()}

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    if args.use_cond_geo:
        assert args.context_node_nf > 0
        context = target_context.unsqueeze(1).unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
        # bs --> bs, 1, 1
    else:
        context = None

    if args.probabilistic_model == 'diffusion':
        if args.use_cond_geo:
            temp_dir = f"temp_dir_gstep{args.guidance_steps}_gscale{args.grad_clip_threshold}_scale{args.clf_scale}_prop{args.property}_use_cond_geo"
        elif args.bilevel_opti:
            temp_dir = f"temp_dir_gstep{args.guidance_steps}_gscale{args.grad_clip_threshold}_fscale{args.clf_scale_force}_pscale{args.clf_scale_prop}_prop_{args.property}_every_{args.every_n_step}_step_K{args.recurrent_times}_method_{args.bilevel_method}"
        elif args.use_evolution:
            temp_dir = f"temp_dir_gstep{args.guidance_steps}_gscale{args.grad_clip_threshold}_scale{args.clf_scale}_every_{args.every_n_step}_recurrent_{args.recurrent_times}_prop{args.property}_package_{args.package}_evo_num_beams{args.num_beams}_interval{args.check_variants_interval}"
        else:
            temp_dir = f"temp_dir_gstep{args.guidance_steps}_gscale{args.grad_clip_threshold}_scale{args.clf_scale}_every_{args.every_n_step}_recurrent_{args.recurrent_times}_prop{args.property}_package_{args.package}"
        if args.cond_neural_model is not None:
            args.cond_neural_model = args.cond_neural_model.to(device)

        guidance_kwargs = {"scale": args.clf_scale,
                           "target": 0.,
                           "use_one_step": True,
                           "guidance_steps": args.guidance_steps,
                           "grad_clip_threshold": args.grad_clip_threshold,
                           "atom_mapping": atom_mapping,
                           "n_cores": args.n_cores,
                           "max_core": args.max_core,
                           "acc": args.acc,
                           "temp_dir": temp_dir,
                           "package": args.package,
                           "every_n_step": args.every_n_step,
                           "use_hist_grad": args.use_hist_grad,
                           "cond_neural_model": args.cond_neural_model,
                           "property": args.property,
                           "run_id": args.run_id,
                           "target_context": target_context,
                           "recurrent_times": args.recurrent_times,
                           "clf_scale_force": args.clf_scale_force,
                           "clf_scale_prop": args.clf_scale_prop
                           }

        print("Guidance kwargs: ", guidance_kwargs)

        x, h = generative_model.cond_sample(batch_size, max_n_nodes, node_mask, edge_mask, context,
                                            fix_noise=fix_noise,
                                            use_neural_guidance=args.use_neural_guidance,
                                            guidance_kwargs=guidance_kwargs,
                                            save_traj_flag=args.save_traj == 1,
                                            use_bilevel=args.bilevel_opti,
                                            args=args)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask


def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist,
                                            nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask

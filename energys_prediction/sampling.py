import torch
from util.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import logging
import os
from qm9.property_prediction import prop_utils
from torch import nn
from qm9.analyze import analyze_stability_for_molecules
from qm9.models import DistributionProperty,DistributionNodes
from energys_prediction.models import EGNN_energy_QM9
from energys_prediction.en_diffusion import EnergyDiffusion
from models_conditional.models import EGNN_dynamics_QM9
from models_conditional.en_diffusion import EnVariationalDiffusion
from util.fingerprint import h_to_charges_qm9
import qm9.visualizer as vis

def get_model(args, argse, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    #in_node_nf: the numbder of atom type
    nodes_dist = DistributionNodes(histogram)
    #distribution for number of atoms: p(M) is categorical distribution, where p(M = n)= number of molecule with n/total samples

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)
    # distribution for property: p(c|M) is categorical distribution,
    #which given the fixed M, divide the [c_min,c_max] into 1000 bins, and count the frequency in each bin.

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    net_energy = EGNN_energy_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=argse.context_node_nf,
        n_dims=3, device=device, hidden_nf=argse.nf,
        act_fn=torch.nn.SiLU(), n_layers=argse.n_layers,
        attention=argse.attention, tanh=argse.tanh, mode=argse.model, norm_constant=argse.norm_constant,
        inv_sublayers=argse.inv_sublayers, sin_embedding=argse.sin_embedding,
        normalization_factor=argse.normalization_factor, aggregation_method=argse.aggregation_method)

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )
        guidance = EnergyDiffusion(
            dynamics=net_energy,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=argse.diffusion_steps,
            noise_schedule=argse.diffusion_noise_schedule,
            noise_precision=argse.diffusion_noise_precision,
            norm_values=argse.normalize_factors,
            include_charges=argse.include_charges
        )
        return vdm, guidance, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)

def sample(args, device, generative_model,guidance,l, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

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

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = None

    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise,guidance=guidance,l=l)

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

loss_l1 = nn.L1Loss()
def test(model, loader, mean, mad, property, device,log_interval=20, dataset_info = None,result_path=None,rdkit=True,save = True):
    model.eval()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    count = 0
    context = []
    for i, data in enumerate(loader):
        molecules['one_hot'].append(data['one_hot'].detach().cpu())
        molecules['x'].append(data['positions'].detach().cpu())
        molecules['node_mask'].append(data['atom_mask'].detach().cpu())

        batch_size, n_nodes, _ = data['positions'].size()
        logging.info(f'generated samples:{(i+1)*batch_size}/10000')
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].to(device, torch.float32)
        nodes = data['one_hot'].to(device, torch.float32)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[property].to(device, torch.float32)
        context.append(label)
        if save:
            node_mask = data['atom_mask'].unsqueeze(2)
            save_path = os.path.join(result_path,'samples')
            os.makedirs(save_path, exist_ok=True)
            charges = h_to_charges_qm9(data['one_hot'].cpu())
            vis.save_xyz_file(save_path, data['one_hot'], charges, data['positions'], dataset_info,
                count, name='samples', node_mask=node_mask)
            count += batch_size


        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)
        loss = loss_l1(mad * pred + mean, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        if i % log_interval == 0:
            logging.info("Iteration %d \t loss %.4f" % (i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))

    torch.save(context, os.path.join(result_path, 'context.pt'))
    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info, rdkit)
    return res['loss'] / res['counter'], stability_dict, rdkit_metrics

def test_bound(model, loader, mean, mad, property, device, log_interval=20):
    model.eval()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].to(device, torch.float32)
        nodes = data['one_hot'].to(device, torch.float32)

        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[property].to(device, torch.float32)

        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        loss = loss_l1(mad * pred + mean, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())


        if i % log_interval == 0:
            print("Iteration %d \t loss %.4f" % ( i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))

    return res['loss'] / res['counter']



from util.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask,check_mask_correct
import utils
import logging
import torch
from qm9.models import DistributionProperty, DistributionNodes
from energys_prediction.models import EGNN_energy_QM9
from energys_prediction.en_diffusion import EnergyDiffusion

def get_model(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_energy = EGNN_energy_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    if args.probabilistic_model == 'diffusion':
        guidance = EnergyDiffusion(
            dynamics=net_energy,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
        )
        return guidance, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)

def compute_loss(prediction_model, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    assert_correctly_masked(x, node_mask)
    # Here x is a position tensor, and h is a dictionary with keys
    # 'categorical' and 'integer'.
    loss = prediction_model(x, h, node_mask, edge_mask,context)
    return loss

def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist,lr_scheduler,partition='train'):
    if partition == 'train':
        lr_scheduler.step()
        model_dp.train()
        model.train()
    else:
        model_ema.eval()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        batch_size, _ , _ = x.size()
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        for key in args.conditioning:
            properties = data[key]
            label = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
            label = label.to(device,dtype)

        if partition == 'train':
            optim.zero_grad()
            # transform batch through flow
            loss = compute_loss(model_dp, x, h, node_mask, edge_mask, label)
            loss.backward()
            if args.clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            else:
                grad_norm = 0.

            optim.step()
        else:
            print('ema')
            loss = compute_loss(model_ema, x, h, node_mask, edge_mask, label)


        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        # Update EMA if enabled.
        if partition == 'train':
            if args.ema_decay > 0:
                ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            logging.info(f" Epoch: {epoch}, iter: {i}/{n_iterations}, "
                         f"Loss {sum(res['loss_arr'][-10:]) / len(res['loss_arr'][-10:]):.4f} "
                        )

        if args.break_train_epoch:
            break
    return res['loss'] / res['counter']
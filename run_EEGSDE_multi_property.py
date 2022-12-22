import argparse
import os
from tool.utils import available_devices,format_devices
device = available_devices(threshold=10000,n_devices=1)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
from os.path import join
import torch
import pickle
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.property_prediction import main_qm9_prop
import os
from tool.utils import set_logger
import logging
from energys_prediction.sampling_multi import get_model,test,sample

def get_classifier(classifiers_path, args_classifiers_path, device='cpu'):
    with open(args_classifiers_path, 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(classifiers_path, map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)
    return classifier

def get_args_gen(args_path, argse_path, argse_path2):
    logging.info(f'args_path:{args_path}')
    with open(args_path, 'rb') as f:
        args_gen = pickle.load(f)
    assert args_gen.dataset == 'qm9_second_half'

    logging.info(f'argse_path1:{argse_path}')
    with open(argse_path, 'rb') as f:
        args_en = pickle.load(f)

    logging.info(f'argse_path2:{argse_path2}')
    print(argse_path2)
    with open(argse_path2, 'rb') as f:
        args_en2 = pickle.load(f)

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    return args_gen, args_en, args_en2


def get_generator(model_path, guidance_path1,guidance_path2,dataloaders, device, args_gen,args_en,args_en2, property_norms):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)

    #model
    model, guidance, guidance2, nodes_dist, prop_dist = get_model(args_gen,args_en, args_en2,device, dataset_info, dataloaders['train'])
    logging.info(f'model_path:{model_path}')
    model_state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_state_dict)

    logging.info(f'energy_path1:{guidance_path1}')
    energy_state_dict = torch.load(guidance_path1, map_location='cpu')
    guidance.load_state_dict(energy_state_dict)

    logging.info(f'energy_path2:{guidance_path2}')
    energy_state_dict2 = torch.load(guidance_path2, map_location='cpu')
    guidance2.load_state_dict(energy_state_dict2)

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), guidance.to(device),guidance2.to(device),nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


class DiffusionDataloader:
    def __init__(self, args_gen, model, guidance,guidance2,l,l2,nodes_dist, prop_dist, device, unkown_labels=False,
                 batch_size=1, iterations=200):
        self.args_gen = args_gen
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = get_dataset_info(self.args_gen.dataset, self.args_gen.remove_h)
        self.i = 0
        self.guidance = guidance
        self.l = l
        self.guidance2 = guidance2
        self.l2 = l2

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,self.guidance,self.l,self.guidance2,self.l2,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)

        node_mask = node_mask.squeeze(2)
        context = context.squeeze(1)

        # edge_mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        context_unorm = {self.prop_dist.properties[0]: [], self.prop_dist.properties[1]: []}
        for i in range(len(self.prop_dist.properties)):
            key = self.prop_dist.properties[i]
            context_i = context[:, i] * self.prop_dist.normalizer[key]['mad'] + self.prop_dist.normalizer[key]['mean']
            context_unorm[key] = context_i

        data = {
            'positions': x.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            'one_hot': one_hot.detach(),
            self.prop_dist.properties[0]: context_unorm[self.prop_dist.properties[0]].detach(),
            self.prop_dist.properties[1]: context_unorm[self.prop_dist.properties[1]].detach()
        }
        return data

    def __next__(self):
        if self.i <= self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


def main_quantitative(args):
    classifier = get_classifier(args.classifiers_path1, args.args_classifiers_path1).to(args.device)
    classifier2 = get_classifier(args.classifiers_path2, args.args_classifiers_path2).to(args.device)

    args_gen, args_en, args_en2 = get_args_gen(args.args_generators_path, args.args_energy_path1, args.args_energy_path2)

    args_gen.load_charges = False
    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, guidance, guidance2, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path, args.energy_path1, args.energy_path2, dataloaders,
                                                    args.device, args_gen, args_en, args_en2, property_norms)

    diffusion_dataloader = DiffusionDataloader(args_gen, model, guidance,guidance2,args.l1,args.l2,nodes_dist, prop_dist,
                                               args.device, batch_size=args.batch_size, iterations=args.iterations)
    test(model=classifier, loader=diffusion_dataloader, property_norms=property_norms, property=args_gen.conditioning, device=args.device, dataset_info=dataset_info, classifier2=classifier2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='eegsde_alpha_mu')
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--generators_path', type=str)
    parser.add_argument('--args_generators_path', type=str)
    parser.add_argument('--energy_path1', type=str)
    parser.add_argument('--args_energy_path1', type=str)
    parser.add_argument('--energy_path2', type=str)
    parser.add_argument('--args_energy_path2', type=str)
    parser.add_argument('--classifiers_path1', type=str)
    parser.add_argument('--args_classifiers_path1', type=str)
    parser.add_argument('--classifiers_path2', type=str)
    parser.add_argument('--args_classifiers_path2', type=str)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)

    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    args.result_path = os.path.join('outputs', args.exp_name, 'l1_' + str(args.l1) + '_' + 'l2_' + str(args.l2))
    os.makedirs(args.result_path, exist_ok=True)
    set_logger(args.result_path, 'logs.txt')
    main_quantitative(args)

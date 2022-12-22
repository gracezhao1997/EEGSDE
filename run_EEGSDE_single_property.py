import argparse
import os
from tool.utils import available_devices,format_devices
device = available_devices(threshold=10000,n_devices=1)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
import torch
import pickle
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.property_prediction import main_qm9_prop
import os
from tool.utils import set_logger,timenow
import logging
from energys_prediction.sampling import sample, test, get_model
from tool.reproducibility import set_seed

def get_classifier(classifiers_path,args_classifiers_path, device='cpu'):
    with open(args_classifiers_path, 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(classifiers_path, map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)
    return classifier

def get_args_gen(args_path,argse_path):
    logging.info(f'args_path:{args_path}')
    with open(args_path, 'rb') as f:
        args_gen = pickle.load(f)
    assert args_gen.dataset == 'qm9_second_half'

    logging.info(f'argse_path:{argse_path}')
    with open(argse_path, 'rb') as f:
        args_en = pickle.load(f)

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    return args_gen,args_en


def get_generator(model_path, guidance_path,dataloaders, device, args_gen,args_en, property_norms):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    model, guidance, nodes_dist, prop_dist = get_model(args_gen,args_en, device, dataset_info, dataloaders['train'])
    logging.info(f'model_path:{model_path}')
    model_state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_state_dict)
    logging.info(f'energy_path:{guidance_path}')
    energy_state_dict = torch.load(guidance_path, map_location='cpu')
    guidance.load_state_dict(energy_state_dict)

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), guidance.to(device),nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


class DiffusionDataloader:
    def __init__(self, args_gen, model, guidance,l,nodes_dist, prop_dist, device, unkown_labels=False,
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

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,self.guidance,self.l,
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

        prop_key = self.prop_dist.properties[0]
        if self.unkown_labels:
            context[:] = self.prop_dist.normalizer[prop_key]['mean']
        else:
            context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
        data = {
            'positions': x.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            'one_hot': one_hot.detach(),
            prop_key: context.detach()
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
    #load property prediction model for evaluation
    classifier = get_classifier(args.classifiers_path,args.args_classifiers_path).to(args.device)

    # args
    args_gen,args_en = get_args_gen(args.args_generators_path,args.args_energy_path)

    #dataloader
    args_gen.load_charges = False
    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    #load conditional EDM and property prediction model
    model, guidance, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path, args.energy_path, dataloaders,
                                                    args.device, args_gen,args_en, property_norms)

    #create a dataloader with EEGSDE
    diffusion_dataloader = DiffusionDataloader(args_gen, model, guidance,args.l,nodes_dist, prop_dist,
                                               args.device, batch_size=args.batch_size, iterations=args.iterations)
    #evaluation
    loss, stability_dict, rdkit_metrics = test(classifier, diffusion_dataloader, mean, mad, args.property, args.device, 1, dataset_info,args.result_path,args.save)
    print("MAE: %.4f" % loss)
    logging.info("MAE: %.4f" % loss)
    logging.info(stability_dict)
    rdkit_metrics = rdkit_metrics[0]
    logging.info("Novelty: %.4f" % rdkit_metrics[2])
    print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='eegsde_mu', help='the name of experiments')
    parser.add_argument('--l', type=float, default=1.0, help='the sacle of guidance')
    parser.add_argument('--property', type=str, default='mu', help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--generators_path', type=str, default='/data/zhaomin/projects/molecular/EDM_single/outputs/exp_cond_mu/generative_model_ema_2020.npy')
    parser.add_argument('--args_generators_path', type=str, default='/data/zhaomin/projects/molecular/EDM_single/outputs/exp_cond_mu/args_2020.pickle')
    parser.add_argument('--energy_path', type=str, default='/data/zhaomin/projects/molecular/EGSDE/outputs/predict_mu/L1_loss/2022-08-05-14-20-45/generative_model_ema_2000.npy')
    parser.add_argument('--args_energy_path', type=str, default='/data/zhaomin/projects/molecular/EGSDE/outputs/predict_mu/L1_loss/2022-08-05-14-20-45/auto_args_2000.pickle')
    parser.add_argument('--classifiers_path', type=str, default='/data/zhaomin/projects/molecular/EDM_single/qm9/property_prediction/outputs/exp_class_mu/best_checkpoint.npy')
    parser.add_argument('--args_classifiers_path', type=str, default='/data/zhaomin/projects/molecular/EDM_single/qm9/property_prediction/outputs/exp_class_mu/args.pickle')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for each iteration')
    parser.add_argument('--iterations', type=int, default=100, help='the number of iterations')
    parser.add_argument('--save', type=str, default=True, help='whether save the generated molecules as .txt')

    #args
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(1234)

    #save path
    args.result_path = os.path.join('outputs', args.exp_name, 'l_' + str(args.l))
    os.makedirs(args.result_path, exist_ok=True)
    set_logger(args.result_path, 'logs.txt')
    main_quantitative(args)

import os.path
import argparse
import torch
from torch import nn

if __name__ == '__main__':
    #Step 3: compute the MAE between properties of generated molecules and desired properties
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', type=str, default='outputs/eegsde_mu/l1.0/gjf_property')
    parser.add_argument('--property', type=str, default='mu')
    parser.add_argument('--label_path', type=str, default='outputs/eegsde_mu/l1.0/context.pt')

    args = parser.parse_args()
    log_root = args.log_root
    fnames = sorted(os.listdir(log_root))
    label = torch.load(args.label_path, map_location='cpu')
    label = torch.hstack(label)

    gts = [] # the condition
    property = []# the computed property by gaussian
    indexs = []

    number = 200
    j = 0
    for fname in fnames:
        if j >= number:
            break
        logpath = os.path.join(log_root, fname)
        with open(logpath, 'r', encoding="ISO-8859-1") as f:
            lines = f.readlines()
            if 'Normal termination of Gaussian' in lines[-1]:
                j += 1
                #the index-th molecule, which can be computed by gaussian
                index = fname.split('.')[0]
                index = index.split('_')[-1]
                index = int(index)
                gt = label[index]
                gts.append(gt)
                indexs.append(index)
                g_property = 0
                for idx, line in enumerate(lines):
                    if args.property == 'mu':
                        if "Dipole moment (field-independent basis, Debye)" in line:
                            words = lines[idx + 1]
                            mu = words.split(' ')[-1]
                            mu = mu.split('\n')[0]
                            g_property = mu
                    if args.property == 'alpha':
                        if "Isotropic polarizability for W=" in line:
                            alpha = line.split(' ')[-2]
                            g_property = alpha
                    if args.property == 'homo':
                        if "Alpha  occ. eigenvalues" in line:
                            homo = line.split(' ')[-1]
                            g_property = homo
                    if args.property == 'lumo':
                        if "Alpha  occ. eigenvalues" in line:
                            words = lines[idx + 1]
                            lumo = words.split('--')[-1]
                            lumo = lumo.lstrip()
                            lumo = lumo.split(' ')[0]
                            g_property = lumo
                    if args.property == 'gap':
                        if "Alpha  occ. eigenvalues" in line:
                            homo = line.split(' ')[-1]
                            words = lines[idx + 1]
                            lumo = words.split('--')[-1]
                            lumo = lumo.lstrip()
                            lumo = lumo.split(' ')[0]
            else:
                #ignore the molecules, which cann't be computed by gaussian
                continue
        # for the lumo, homo and gap, they should multiply 27.2114 to be consistent with QM9
        if args.property == 'homo' or args.property == 'lumo':
            g_property = float(g_property) * 27.2114
        elif args.property == 'gap':
            gap = float(lumo) - float(homo)
            g_property = gap * 27.2114
        else:
            g_property = float(g_property)
        property.append(torch.tensor([g_property]))
    gts = torch.hstack(gts)
    property = torch.hstack(property)
    loss = nn.L1Loss()
    MAE = loss(gts, property)
    print('log_root:', args.log_root)
    print('MAE:', MAE)










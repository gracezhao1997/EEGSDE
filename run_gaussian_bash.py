import os
import argparse

def get_name(fname):
    name = os.path.split(fname)[1]
    name = os.path.splitext(name)[0]
    return name
if __name__ == '__main__':
    #Step2: use the Gaussian software to compute the generated molecules
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_root', type=str, default='outputs/eegsde_mu/l1.0/gjf')
    parser.add_argument('--save_root', type=str, default='outputs/eegsde_mu/l1.0/gjf_property')
    args = parser.parse_args()

    fnames = sorted(os.listdir(args.samples_root))
    os.makedirs(args.save_root, exist_ok=True)

    for fname in fnames:
        print(fname)
        name = get_name(fname)
        sample_path = os.path.join(args.samples_root, name+'.gjf')
        save_path = os.path.join(args.save_root, name+'.log')
        os.system('$g16root/g16/g16 ' + sample_path + ' ' + save_path)

# #step3: evalution
# property = 'lumo'
# samples_root = 'outputs/gaussian_property/egsde_'+property
# dirs = os.listdir(samples_root)
# for dir in dirs:
#     log_root = os.path.join(samples_root,dir,'gjf_property')
#     label_path = os.path.join(samples_root,dir, 'context.pt')
#     os.system('python run_evaluate_gaussian.py --log_root ' + log_root + ' --property ' + property + ' --label_path '+ label_path)




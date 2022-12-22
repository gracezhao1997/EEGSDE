import os.path
import argparse
def get_name(fname):
    name = os.path.split(fname)[1]
    name = os.path.splitext(name)[0]
    return name


def process(fname, save_fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    name = get_name(fname)
    res = f'%nprocshared=8\n'
    res +=  f'%mem=16GB\n'
    res += f'%chk={name}.chk\n'
    res += '# opt freq b3lyp 6-31G(2df,p)\n\nTitle Card Required\n\n'
    res += f'0 1\n'
    num_atoms = int(lines[0])
    for line in lines[2: 2 + num_atoms]:
        typ, x, y, z = line.split(' ')[:4]
        res += f' {typ}                 {x[:-1]}    {y[:-1]}    {z[:-1]}\n'
    res += '\n\n'

    with open(save_fname, 'w') as f:
        f.write(res)

if __name__ == '__main__':
    # Step 1: convert_gjf.py to convert the .txt file, which save the generated molecules, to .gjf file for Gaussian software
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_root', type=str, default='outputs/eegsde_mu/l1.0/samples')
    parser.add_argument('--gif_root', type=str, default='outputs/eegsde_mu/l1.0/gjf')
    args = parser.parse_args()

    os.makedirs(args.gif_root, exist_ok=True)
    fnames = sorted(os.listdir(args.samples_root))
    for fname in fnames:
        if fname.endswith('.txt'):
            name = get_name(fname)
            process(os.path.join(args.samples_root, fname), os.path.join(args.gif_root, f'{name}.gjf'))







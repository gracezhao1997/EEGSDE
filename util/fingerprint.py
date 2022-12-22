import torch
import numpy as np
import openbabel as ob
import pybel
from ase.data import atomic_masses

def compute_fingerprint(poss,numberss,num_atomss):
    fingerprint = []
    ids = len(num_atomss)
    for i in range(ids):
        pos = poss[i, :]
        numbers = numberss[i, :].squeeze()
        num_atoms = num_atomss[i]

        numbers = numbers[:num_atoms]
        pos = pos[:num_atoms]

        # minius compute mass
        m = atomic_masses[numbers]
        com = np.dot(m, pos) / m.sum()
        pos = pos - com

        # order atoms by distance to center of mass
        d = torch.sum(pos ** 2, dim=1)
        center_dists = torch.sqrt(torch.maximum(d, torch.zeros_like(d)))
        idcs_sorted = torch.argsort(center_dists)
        pos = pos[idcs_sorted]
        numbers = numbers[idcs_sorted]

        # Open Babel OBMol representation
        obmol = ob.OBMol()
        obmol.BeginModify()
        # set positions and atomic numbers of all atoms in the molecule
        for p, n in zip(pos, numbers):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.EndModify()
        _fp = pybel.Molecule(obmol).calcfp()
        fp = np.array(_fp.fp, dtype=np.uint32)
        # convert fp to 1024bit
        fp = np.array(fp, dtype='<u4')
        fp = torch.FloatTensor(
            np.unpackbits(fp.view(np.uint8), bitorder='little'))
        fingerprint.append(fp)
    return fingerprint

def compute_fingerprint_bits(poss,numberss,num_atomss):
    fingerprint_1024 = []
    fingerprint_bits = []
    ids = len(num_atomss)
    for i in range(ids):
        pos = poss[i, :]
        numbers = numberss[i, :].squeeze()
        num_atoms = num_atomss[i]

        numbers = numbers[:num_atoms]
        pos = pos[:num_atoms]

        # minius compute mass
        m = atomic_masses[numbers]
        com = np.dot(m, pos) / m.sum()
        pos = pos - com

        # order atoms by distance to center of mass
        d = torch.sum(pos ** 2, dim=1)
        center_dists = torch.sqrt(torch.maximum(d, torch.zeros_like(d)))
        idcs_sorted = torch.argsort(center_dists)
        pos = pos[idcs_sorted]
        numbers = numbers[idcs_sorted]

        # Open Babel OBMol representation
        obmol = ob.OBMol()
        obmol.BeginModify()
        # set positions and atomic numbers of all atoms in the molecule
        for p, n in zip(pos, numbers):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.EndModify()
        _fp = pybel.Molecule(obmol).calcfp()
        fp_bits = {*_fp.bits}
        fingerprint_bits.append(fp_bits)

        fp_32 = np.array(_fp.fp, dtype=np.uint32)
        # convert fp to 1024bit
        fp_1024 = np.array(fp_32, dtype='<u4')
        fp_1024 = torch.FloatTensor(
            np.unpackbits(fp_1024.view(np.uint8), bitorder='little'))
        fingerprint_1024.append(fp_1024)

    return fingerprint_bits,fingerprint_1024

def tanimoto(fp1_bitss,fp2_bitss):
    s_sum = 0.0
    bs = len(fp1_bitss)
    for i in range(bs):
        fp1_bits = fp1_bitss[i]
        fp2_bits = fp2_bitss[i]
        n_equal = len(fp1_bits.intersection(fp2_bits))
        if len(fp1_bits) + len(fp2_bits) == 0:  # edge case with no set bits
            s = 1.0
        else:
            s = n_equal / (len(fp1_bits)+len(fp2_bits)-n_equal)
        s_sum += s
    return s_sum/bs

def h_to_charges_qm9(one_hots):
    # one_hot = one_hots[:,0]
    bs,n_nodes,_ = one_hots.size()
    charges = torch.zeros(bs,n_nodes,dtype=torch.int64)
    name = torch.tensor([1,6,7,8,9],dtype=torch.int64)
    index = (one_hots == 1).nonzero(as_tuple=True)
    charge  = torch.index_select(name, 0, index[2])
    charges[index[0],index[1]] = charge
    return charges.unsqueeze(2)

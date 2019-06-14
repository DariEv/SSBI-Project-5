from Bio.PDB import *
import numpy as np

from scipy.constants import pi
from scipy.constants import epsilon_0

__q_plus__ = 0.42
__q_minus__ = -0.2
__threshold__ = -2.4 


## todo delete later 
## todo irregular aa are deleted ? 
AMINO_ACIDS = ['ALA', 'GLY', 'PHE', 'ILE', 'MET', 'LEU', 'PRO', 'VAL', 'ASP', 'GLU', 'LYS', 'ARG', 'SER', 'THR', 'TYR',
               'HIS', 'CYS', 'ASN', 'GLN', 'TRP']


def import_pdb():
    
    p = PDBParser()
    structure = p.get_structure('X', 'supplementary/1b35.pdb')
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in AMINO_ACIDS:
                    residues.append(residue)
        break
    return residues


def dssp(residues, h_coord):
    
    n_res = len(residues)
    print("n", n_res)
    print("n_H", len(h_coord))
    energy = pairwise_e(residues, h_coord)
    #print(energy[0])
    #print(energy[2])
    h_bonds = np.zeros(n_res)
    
    for i in range(n_res):
        j = np.argmin(energy[i])
        #print("w", j)
        if energy[i][j] < __threshold__:
            h_bonds[i] = j
            #h_bonds[j] = i+1
            
    print(h_bonds)
    
    return h_bonds


def pairwise_e(residues, h_coord):
    
    #print(h_coord[0], "----")
    #print(np.array((list(residues[1]["N"].get_vector()))), "3333")
    #print(calculate_distance(residues[1]["N"], h_coord[0]))
    
    # todo filter out het and HOH
    n_res = len(residues)
    energy = np.zeros((n_res, n_res))
    
    for i in range(n_res):
        for j in range(i + 1, n_res):
            
            res1 = residues[i]
            res2 = residues[j]
            atoms_present = (res1.has_id("O") or res1.has_id("C")) and res2.has_id("N")
            
            if atoms_present:
                r_on = res1["O"] - res2["N"]
                r_ch = calculate_distance(res1["C"], h_coord[j-1])
                r_oh = calculate_distance(res1["O"], h_coord[j-1])
                r_cn = res1["C"] - res2["N"]
                energy[i][j] = get_energy(r_on, 1,1, r_cn)
                energy[j][i] = energy[i][j]
                #print("d", r_on, r_cn)

    return energy


def get_energy(r_on, r_ch, r_oh, r_cn):
    
    return (__q_plus__*__q_minus__*(1/r_on + 1/r_ch - 1/r_oh - 1/r_cn))/(4*pi*epsilon_0)
    

def calculate_distance(atom, h_atom):
    p1 = np.array((list(atom.get_vector())))
    p2 = h_atom
    
    return np.linalg.norm(p1-p2)

# The program starts here:

def main():
    print("DSSP")
    #res = import_pdb()
    #print(pairwise_e(res))
    #print(dssp(res))
    

if __name__ == "__main__":
    main()
    
    
    
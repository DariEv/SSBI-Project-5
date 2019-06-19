from Bio.PDB import *
import numpy as np
import os
import collections

import h_bonds_calculator
import ssbi_project_h_atoms


WINDOW_SIZE = 10


AMINO_ACIDS = ['ALA', 'GLY', 'PHE', 'ILE', 'MET', 'LEU', 'PRO', 'VAL', 'ASP', 'GLU', 'LYS', 'ARG', 'SER', 'THR', 'TYR',
               'HIS', 'CYS', 'ASN', 'GLN', 'TRP']


class FeatureExtractor:
    
    def __init__(self, p = ""):
        self.path2dir = p
        
        
    def get_features(self):
        
        all_features = []
        
        #  iterate through files
        
        for f in os.listdir(self.path2dir):
            
            residues = self.parse_pdb_file(f)
                        
            # iterate through residues 
            features, h_coords = self.get_initial_features(residues)
            #print(features)
            #print(h_coords)
            
            h_bonds = h_bonds_calculator.get_bonds(residues, h_coords) 
            #print(h_bonds)
            
            env_feature = self.get_environment_features(h_bonds)
            print(env_feature)
            
            # add H-bond to the end of feautre vector
            features = list(zip(features, h_bonds))
            features = [i[0] + [i[1]] for i in features]
            #print(features)
            
            all_features = all_features + features
            
        #print(all_features)
        print(len(all_features))
        
        return all_features
            
            
    def parse_pdb_file(self, file):
        
        pdb_parser = PDBParser(QUIET=True)
        name = os.path.basename(file).split('.')[0]
        structure = pdb_parser.get_structure(name, self.path2dir + file)
    
        residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in AMINO_ACIDS:
                        residues.append(residue)
            break
        
        return residues
    
    
    def get_initial_features(self, residues):
        
        # save the H coordinates in list
        h_coord_per_res = []
        
        # save all features in list
        features_per_res = []
        
        # Extract Amino Acid and Amino Acid one later
        for k, amino_acid in enumerate(residues):
            if k == len(residues)-1:
                break
            amino_acid1_ats = list(amino_acid.get_atoms())
            o_coord = None
            c_coord = None
            for at in amino_acid1_ats:
                if at.get_name() == 'O':
                    o_coord = at.get_coord()
                if at.get_name() == 'C':
                    c_coord = at.get_coord()
    
                if c_coord is not None and o_coord is not None:
                    break
    
            amino_acid2_ats = list(residues[k+1].get_atoms())
            c_alpha_coord = None
            n_coord = None
            for at in amino_acid2_ats:
                if at.get_name() == 'N':
                    n_coord = at.get_coord()
    
                if at.get_name() == 'CA':
                    c_alpha_coord = at.get_coord()
    
                if c_alpha_coord is not None and n_coord is not None:
                    break
    
    
            phi_angle = dihedral_angle(residues[k]['C'].get_coord(), residues[k+1]['N'].get_coord(),
                                       residues[k+1]['CA'].get_coord(), residues[k+1]['C'].get_coord())
    
            psi_angle = dihedral_angle(residues[k]['N'].get_coord(), residues[k]['CA'].get_coord(),
                                       residues[k]['C'].get_coord(), residues[k+1]['N'].get_coord())
    
            #print([phi_angle, psi_angle, ssbi_project_h_atoms.calculate_h(np.array(o_coord), np.array(c_coord), np.array(c_alpha_coord),
             #                                       np.array(n_coord))])
    
            features_per_res.append([phi_angle, psi_angle])
            
            h_coord_per_res.append(ssbi_project_h_atoms.calculate_h(np.array(o_coord), 
                                                                    np.array(c_coord), 
                                                                    np.array(c_alpha_coord), 
                                                                    np.array(n_coord)))
            
        return features_per_res, h_coord_per_res
    
    
    def get_environment_features(self, h_bonds):
        
        n = len(h_bonds)
        env_features = []
        
        for i, bond in enumerate(h_bonds):
            
            left_range = i - WINDOW_SIZE
            right_range = i + WINDOW_SIZE
            
            left_offset = []
            right_offset = []
            
            if left_range < 0:
                left_offset = list(np.full(abs(left_range), np.nan))
                print(left_offset)
                left_range = 0
            if right_range > n:
                right_offset = list(np.full(right_range, np.nan))
                print(right_offset)
                right_range = n - 1
                
            env = h_bonds[left_range : right_range]
                
            diversity = len(collections.Counter(env))
            
            env_features.append([env, diversity])
            
        return env_features

# TODO: replace with PDB function                                           
# Calculate dihedral angle
def dihedral_angle(c1, c2, c3, c4):

    # Compute the three vectors spanning the angle
    v1 = (c2 - c1) * -1
    v2 = c3 - c2
    v3 = c4 - c3

    # Compute cross products
    v1_v2 = np.cross(v1, v2)
    v2_v3 = np.cross(v3, v2)

    v1v2_x_v2v3 = np.cross(v1_v2, v2_v3)

    # Compute the degree of the angle
    angle = np.degrees(np.arctan2((np.dot(v1v2_x_v2v3, v2) * (1.0/np.linalg.norm(v2))), np.dot(v1_v2, v2_v3)))
    return (angle)                                                
    

# testing 
fe = FeatureExtractor("supplementary_small/")
print(fe.path2dir)
fe.get_features()






    
        
    
from Bio.PDB import *
import numpy as np
import os

import h_bonds_calculator
import ssbi_project_h_atoms


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

            parsed_structures = self.parse_structures(self.path2dir+f)

            structures = self.get_structures(residues, parsed_structures)
                        
            # iterate through residues

            features, h_coords = self.get_initial_features(residues)
            #print(features)
            #print(h_coords)
            
            h_bonds = h_bonds_calculator.get_bonds(residues, h_coords) 
            #print(h_bonds)
            
            # feature vector: [phi, psi, distance (h-bonds), structure]
            features = list(zip(features, h_bonds, structures))
            features = [i[0] + [i[1]] + [i[2]] for i in features]
            
            all_features = all_features + features
            
        print(all_features)
        
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


    def parse_structures(self, file):

        with open(file) as f:

            parsed_structures = []

            temp = ''
            for line in f:
                if line[:6] == 'HELIX ':
                    helix_type = int(line[38:40])
                    if helix_type == 1:     # alpha
                        type = 1
                    elif helix_type == 5:   # 310
                        type = 2
                    elif helix_type == 3:   # pi
                        type = 3
                    for i in range(int(line[21:25]), int(line[33:37]) + 1):
                        parsed_structures.append([i, line[19], type])
                if line[:6] == 'SHEET ':
                    sheet_type = int(line[38:40])
                    if sheet_type == 1:     # parallel sheet
                        type = 4
                    elif sheet_type == -1:  # anti-parallel sheet
                        type = 5
                    if sheet_type == 0:
                        temp = line
                    else:
                        if temp != '':
                            for i in range(int(temp[22:26]), int(temp[33:37]) + 1):
                                parsed_structures.append([i, temp[21], type])
                            temp = ''
                        for i in range(int(line[22:26]), int(line[33:37]) + 1):
                            parsed_structures.append([i, line[21], type])

        return parsed_structures


    def get_structures(self, residues, parsed_structures):

        structures = []

        for k, amino_acid in enumerate(residues):

            if k == len(residues) - 1:
                break

            feature_vector = []

            # print(amino_acid.get_id()[1])
            aas_helix_sheet = [a[0] for a in parsed_structures]
            # print(aas_helix_sheet)
            if residues[k].get_id()[1] in aas_helix_sheet:
                res = []
                # res = [p for p in parsed_structures if int(p[0]) == int(amino_acid.get_id()[1]) and p[1] == amino_acid.get_parent().get_id()]
                for p in parsed_structures:
                    if int(p[0]) == int(residues[k].get_id()[1]):
                        res.append(p)
                # print(amino_acid.get_parent().get_id())
                if len(res) > 0:
                    structures.append(res[0][2])
            else:
                structures.append(0)

        return structures
    
    
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

    
            features_per_res.append([phi_angle, psi_angle])
            
            h_coord_per_res.append(ssbi_project_h_atoms.calculate_h(np.array(o_coord), 
                                                                    np.array(c_coord), 
                                                                    np.array(c_alpha_coord), 
                                                                    np.array(n_coord)))
            
        return features_per_res, h_coord_per_res

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

    #angle = calc_dihedral(c1, c2, c3, c4)

    return (angle)                                                


# testing 
fe = FeatureExtractor("supplementary_small/")
print(fe.path2dir)
fe.get_features()
    





    
        
    
from Bio.PDB import *
import numpy as np
import os
import collections
import sys
import pickle

import h_bonds_calculator
import ssbi_project_h_atoms

# Feature Vector (68 values):
# Amino Acid (20 values),
# Amino Acid Environment (Avg, 20 values),
# pI (1 value), hydro phobicity (1 value)
# Phy (1 value), Psi (1 value),
# H-Bond (1 value),
# Env (2*WINDOW_SIZE+1),
# diversity (1 value),
# segment length (1 value)

WINDOW_SIZE = 10


AMINO_ACIDS = ['ALA', 'GLY', 'PHE', 'ILE', 'MET', 'LEU', 'PRO', 'VAL', 'ASP', 'GLU', 'LYS', 'ARG', 'SER', 'THR', 'TYR',
               'HIS', 'CYS', 'ASN', 'GLN', 'TRP']


class FeatureExtractor:
    
    def __init__(self, p = ""):
        self.path2dir = p

        # Variables for normalization necessary
        self.min_pI = 2.77
        self.max_pI = 10.76

        self.min_h_phob = -2.53
        self.max_h_phob = 1.38

        self.min_phi_psi = -180
        self.max_phi_psi = 180

        self.min_h_bond = np.inf
        self.max_h_bond = -np.inf

        self.min_env = np.inf
        self.max_env = -np.inf

        self.min_diversity = np.inf
        self.max_diversity = -np.inf

        self.min_seg_length = np.inf
        self.max_seg_length = -np.inf
        #######################################

        self.filter_dict = {'No Residues':[], 'Missing H':[], 'Missing Atoms':[], 'H Bond Error':[]}
        
        self.features, self.labels_q6, self.labels_q3, self.peptide_lengths = self.get_features()

        self.normalized_features = self.__normalize()

        
    def get_features(self):

        all_features = []
        all_structures_q6 = []
        peptide_lengths = []


        file_number = 1
        #  iterate through files
        for f in os.listdir(self.path2dir):

            print('Currently processing file: {}, ({}/200)'.format(f,file_number))
            file_number += 1

            # list of all residues
            residues = self.parse_pdb_file(f)

            ############### Get Files without residues
            if not residues:
                self.filter_dict['No Residues'].append(f)
            ##########################################

            # Check if there is aas in the file
            if residues:
                # get structures given in the PDB file
                parsed_structures = self.parse_structures(self.path2dir+f)
                structures = self.get_structures(residues, parsed_structures)

                # encode the residues in 20-bit vectors
                aas_init = self.aas_init(residues)

                aas_init = self.aa_environment(aas_init)

                ##############################################
                try:
                    angles, h_coords = self.get_initial_features(residues)
                except TypeError:
                    print('Cannot calculate Hydrogens for all residues, skipping file:',f)
                    self.filter_dict['Missing H'].append(f)
                    continue
                except KeyError:
                    # Bsp: 1to2.pdb missing N somewhere!!!
                    print('Missing atoms in structure -> Missing angles, skipping file:',f)
                    self.filter_dict['Missing Atoms'].append(f)
                    continue

                try:
                    h_bonds = h_bonds_calculator.get_bonds(residues, h_coords)
                except ValueError:
                    print('operands could not be broadcast together with shapes (3,) (0,), skipping file:', f)
                    self.filter_dict['H Bond Error'].append(f)
                    continue


                # get min and max h_bond value
                for bond in h_bonds:
                    if np.max(bond) > self.max_h_bond:
                        self.max_h_bond = np.max(bond)

                    if np.min(bond) < self.min_h_bond:
                        self.min_h_bond = np.min(bond)
                ##############################

                envs, diversities, seg_lengths = self.get_environment_features(h_bonds)

                # get min max envs, divs and seg_lengths
                assert len(envs) == len(seg_lengths) and len(seg_lengths) == len(diversities)
                for i in range(0, len(envs)):
                    if np.max(envs[i]) > self.max_env:
                        self.max_env = np.max(envs[i])

                    if np.min(envs[i]) < self.min_env:
                        self.min_env = np.min(envs[i])

                    if np.max(diversities[i]) > self.max_diversity:
                        self.max_diversity = np.max(diversities[i])

                    if np.min(diversities[i]) < self.min_diversity:
                        self.min_diversity = np.min(diversities[i])

                    if np.max(seg_lengths[i]) > self.max_seg_length:
                        self.max_seg_length = np.max(seg_lengths[i])

                    if np.min(seg_lengths[i]) < self.min_seg_length:
                        self.min_seg_length = np.min(seg_lengths[i])
                ######################################################

                # feature vector:
                #[encoded residue name, isoelectric point (pI), hydrophobicity, phi, psi, distance (h-bonds), structure]
                #features = list(zip(aas_init, features, h_bonds, envs, diversities))
                #features = [i[0] + i[1] + [i[2]] + [i[3]] + [i[4]] for i in features]

                features = []
                for i in range(0,len(aas_init)):
                    # Normal features
                    tmp_vec = np.append(np.concatenate((aas_init[i], np.array(angles[i])),axis=0),h_bonds[i])
                    tmp_vec = np.concatenate((tmp_vec, np.array(envs[i])),axis=0)
                    tmp_vec = np.append(tmp_vec, diversities[i])
                    tmp_vec = np.append(tmp_vec, seg_lengths[i])
                    features.append(tmp_vec)

                all_features = all_features + features
                all_structures_q6 = all_structures_q6 + structures
                peptide_lengths.append(sum(peptide_lengths) + len(all_features))

            
        #print(len(all_features))
        #print(len(all_structures_q6))

        all_structures_q3 = []
        for struct in all_structures_q6:
            if struct in [1,2,3]:
                all_structures_q3.append(1)
            elif struct in [4,5]:
                all_structures_q3.append(2)
            else:
                all_structures_q3.append(struct)

        return all_features, all_structures_q6, all_structures_q3, peptide_lengths


    def aa_environment(self, aas_init):
        '''
        Expects a list of feature vectors, where each vector contains the 20 bit AA encoded features and the
        hydrophobicity and iso-electric point
        :param aas_init:
        :return: average neighborhood
        '''

        aas = []

        for i, amino in enumerate(aas_init):
            if i >= WINDOW_SIZE:
                window = aas_init[i-WINDOW_SIZE:i+WINDOW_SIZE+1]
            else:
                window = aas_init[:i+WINDOW_SIZE+1]

            sum = np.zeros((20))
            for aa in window:
                sum = np.add(sum, aa[:20])

            avg = sum/len(window)
            aas.append(np.concatenate((np.append(amino[:20],avg),amino[20:])))

        return aas


    def aas_init(self, residues):

        aas_init = []

        for residue in [r.get_resname() for r in residues]:

            aacode = []
            pI = 0
            h_phob = 0

            if residue == 'ALA':
                aacode = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 6.00
                h_phob = 0.62
            elif residue == 'GLY':
                aacode = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 5.97
                h_phob = 0.48
            elif residue == 'PHE':
                aacode = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 5.48
                h_phob = 1.19
            elif residue == 'ILE':
                aacode = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 6.02
                h_phob = 1.38
            elif residue == 'MET':
                aacode = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 5.74
                h_phob = 0.64
            elif residue == 'LEU':
                aacode = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 5.98
                h_phob = 1.06
            elif residue == 'PRO':
                aacode = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 6.30
                h_phob = 0.12
            elif residue == 'VAL':
                aacode = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 5.96
                h_phob = 1.08
            elif residue == 'ASP':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 2.77
                h_phob = -0.90
            elif residue == 'GLU':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 3.22
                h_phob = -0.74
            elif residue == 'LYS':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 9.74
                h_phob = -1.50
            elif residue == 'ARG':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                pI = 10.76
                h_phob = -2.53
            elif residue == 'SER':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                pI = 5.68
                h_phob = -0.18
            elif residue == 'THR':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                pI = 5.60
                h_phob = -0.05
            elif residue == 'TYR':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                pI = 5.66
                h_phob = 0.26
            elif residue == 'HIS':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                pI = 7.59
                h_phob = -0.40
            elif residue == 'CYS':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                pI = 5.07
                h_phob = 0.29
            elif residue == 'ASN':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                pI = 5.41
                h_phob = -0.78
            elif residue == 'GLN':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                pI = 5.65
                h_phob = -0.85
            elif residue == 'TRP':
                aacode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                pI = 5.89
                h_phob = 0.81

            #aas_init.append(aacode + [pI] + [h_phob])
            aas_init.append(np.append(np.append(np.array(aacode), pI), h_phob))

        return aas_init
            
            
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
                if line[:6] == 'ENDMDL':  # stop after the first model of the structure
                    break
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
                    # Check if character for sheet type exists if not set sheet type to 99 and catch case below
                    if line[38:40].strip() is not '':
                        sheet_type = int(line[38:40])
                    else:
                        sheet_type = 99

                    if sheet_type == 1:     # parallel sheet
                        type = 4
                    elif sheet_type == -1:  # anti-parallel sheet
                        type = 5
                    if sheet_type == 0:
                        temp = line
                    # Maybe add special catch case...
                    # elif sheet_type == 99:
                    #     pass
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

            aas_helix_sheet = [a[0] for a in parsed_structures]

            if amino_acid.get_id()[1] in aas_helix_sheet:
                res = [p for p in parsed_structures if int(p[0]) == int(amino_acid.get_id()[1])]
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

            if k == 0:
                features_per_res.append([phi_angle, psi_angle])
            features_per_res.append([phi_angle, psi_angle])


            h_coord_per_res.append(ssbi_project_h_atoms.calculate_h(np.array(o_coord), 
                                                                    np.array(c_coord), 
                                                                    np.array(c_alpha_coord), 
                                                                    np.array(n_coord)))
            
        return features_per_res, h_coord_per_res
    
    
    def get_environment_features(self, h_bonds):
        
        n = len(h_bonds)
        
        # env = verctor of h_bond distances around the residue
        # diversities = number of different h_bond distances around the residue
        # segment_lengths = sum of distances to the left and right 0's
        
        envs = []  
        diversities = []
        segment_lengths = []
        
        for i, bond in enumerate(h_bonds):
            
            left_range = i - WINDOW_SIZE - 1
            right_range = i + WINDOW_SIZE
            
            left_offset = []
            right_offset = []
            
            
            if left_range < 0:
                left_offset = np.zeros(abs(left_range))            
                left_range = 0
            if right_range > n:
                right_offset = np.zeros(right_range - n + 1)               
                right_range = n - 1
                
            env = np.append(left_offset, h_bonds[left_range : right_range])
            env = np.append(env, right_offset)
                
            diversity = len(collections.Counter(env))
            
            # legment length feature
            segment_length = 0
            
            if bond != 0:

                # to the left 
                counter = i
                while counter > 0 and h_bonds[counter] != 0:
                    segment_length = segment_length + 1
                    counter = counter - 1
                # to the right 
                counter = i
                while counter < n and h_bonds[counter] != 0:
                    segment_length = segment_length + 1
                    counter = counter + 1
                    
            segment_lengths.append(segment_length)
            envs.append(env)
            diversities.append(diversity)
            
        return envs, diversities, segment_lengths

    def __normalize(self):
        norm_feats = []
        for feature in self.features:
            norm_feats.append(self.__normalize_feature(feature))

        return norm_feats

    def __normalize_feature(self, feat):
        feat_vec = feat[:40]
        feat_vec = np.append(feat_vec, np.interp(feat[40:41], [self.min_pI, self.max_pI], [0,1]))
        feat_vec = np.append(feat_vec, np.interp(feat[41:42], [self.min_h_phob, self.max_h_phob], [0,1]))
        feat_vec = np.concatenate((feat_vec, np.interp(feat[42:44], [self.min_phi_psi, self.max_phi_psi], [0,1])), axis=0)
        feat_vec = np.append(feat_vec, np.interp(feat[44:45], [self.min_h_bond, self.max_h_bond], [0,1]))
        feat_vec = np.concatenate((feat_vec, np.interp(feat[45:-2], [self.min_env, self.max_env], [0,1])), axis=0)
        feat_vec = np.append(feat_vec, np.interp(feat[-2], [self.min_diversity, self.max_diversity], [0,1]))
        feat_vec = np.append(feat_vec, np.interp(feat[-1], [self.min_seg_length, self.max_seg_length], [0,1]))
        return feat_vec


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

    return angle


def main():  
    
    input_file = "supplementary_small/"
    #input_file = "supplementary/"
    output_file = "Extracted_Features_small.pkl"
    #output_file = "Extracted_Features.pkl"

# =============================================================================
#     with open(output_file, 'wb') as output:
#
#         fe = FeatureExtractor(input_file)
#         print("Extracting features for the files in:", fe.path2dir)
#         print("Write results in:",output_file)
#         pickle.dump(fe, output, pickle.HIGHEST_PROTOCOL)
# =============================================================================
        
        
    # open saved file
    
    with open(output_file, 'rb') as input:
        saved_features = pickle.load(input)
        print(saved_features.features[0])
        print(saved_features.normalized_features[0])
        missed_files = 0
        for key, value in saved_features.filter_dict.items():
            missed_files += len(value)
        print('#Missed files:',missed_files)
        #print(saved_features.labels_q6)
        #print(saved_features.labels_q3)
   
    
if __name__ == '__main__':
    main()

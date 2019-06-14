from Bio.PDB import *
import os
import numpy as np
import dssp


AMINO_ACIDS = ['ALA', 'GLY', 'PHE', 'ILE', 'MET', 'LEU', 'PRO', 'VAL', 'ASP', 'GLU', 'LYS', 'ARG', 'SER', 'THR', 'TYR',
               'HIS', 'CYS', 'ASN', 'GLN', 'TRP']

ATOMS = ['C', 'N', 'CA', 'O']


def solve_quadratic(a, b, c):
    x_1 = (-b + np.sqrt(b**2 - 4 * a * c))/(2*a)
    x_2 = (-b - np.sqrt(b**2 - 4 * a * c))/(2*a)
    return [x_1,x_2]


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def calculate_h(o, c, ca, n):
    dist_h_n = 0.97

    # Plane defined by O-C-N(-H)
    # Calculate vectors to define plane
    vec_o_c = np.subtract(o,c)
    vec_n_c = np.subtract(n,c)

    # Calculate plane normal
    normal_1 = np.cross(vec_o_c, vec_n_c)

    # plane in coordinateform -> [a,b,c,d] -> ax1 + bx2 + cx3 - d = 0
    plane_1 = np.append(normal_1,np.dot(normal_1, o))

    ###################################################################################
    # Plane defined by normal given through cylindrical calculation
    vec_c_alpha_n = np.subtract(n, ca)
    dist_c_alpha_n = np.linalg.norm(vec_c_alpha_n)

    # Distance between n and n_prime on a second plane
    dist_n_n_prime = abs(np.cos(np.radians(61))*dist_h_n)
    dist_c_alpha_n_prime = dist_c_alpha_n + dist_n_n_prime

    # vector between c alpha and n can be interpreted as normal vector to second plane
    normal_2 = (vec_c_alpha_n/dist_c_alpha_n)*dist_c_alpha_n_prime

    # c_prime is support vector to plane
    n_prime = np.add(ca, normal_2)

    plane_2 = np.append(normal_2,np.dot(n_prime,normal_2))

    #print(plane_1)
    #print(plane_2)

    ###################################################################################
    # solve linear system
    system = np.array([normal_1, normal_2, np.cross(normal_1,normal_2)])
    solution = np.array([np.dot(normal_1,o),np.dot(n_prime,normal_2),0])

    # get point on line -> defines line with direction vector
    support_point = np.linalg.solve(system,solution)
    #print(support_point)

    # Check if == 0 -> point lies in plane -> CAREFUL: NOT ALWAYS EXACTLY ZERO DUE TO ROUNDING ERRORS
    #print(np.dot(plane_1[:3],support_point)-plane_1[3])
    #print(np.dot(plane_2[:3],support_point)-plane_2[3])

    # direction vector of line -> cross between plane normals
    # Line between planes therefore fully defined by direction_vec and support_point!
    direction_vec = np.cross(normal_1, normal_2)

    # Calculating possible points with quadratic solver: ax^2 + bx + m = 0
    a = np.sum(direction_vec**2)
    b = 2*np.dot(support_point, direction_vec)-2*np.dot(direction_vec, n)
    m = np.sum(support_point**2) + np.sum(n**2) - 2 * np.dot(n, support_point) - dist_h_n**2

    # Calculate stretch factors for direction vector
    solutions = None
    if not a == 0:
        solutions = solve_quadratic(a,b,m)
    elif not b == 0:
        solutions = [-m/b]

    #print('Length direction vector: {}'.format(np.linalg.norm(direction_vec)))
    # print(solutions)

    h_atoms = []
    if solutions is not None:
        if len(solutions) == 2:
            point_1 = support_point + solutions[0] * direction_vec
            point_2 = support_point + solutions[1] * direction_vec

            ######## Check if points lie in planes -> 0
            # print(np.dot(plane_1[:3], point_1)-plane_1[3])
            # print(np.dot(plane_2[:3], point_1)-plane_2[3])
            # print(np.dot(plane_1[:3], point_2) - plane_1[3])
            # print(np.dot(plane_2[:3], point_2) - plane_2[3])
            # print()

            ######### Check Distance to N atom -> should be ~0.97A
            # print('Distance to N point_1: {}'.format(np.linalg.norm(np.subtract(n, point_1))))
            # print('Distance to N point_2: {}'.format(np.linalg.norm(np.subtract(n, point_2))))
            # print()

            ######### Check if angle == ~119°
            # print(180-(180*angle(np.subtract(point_1, n), np.subtract(n, ca))/np.pi))
            # print(180-(180*angle(np.subtract(point_2, n), np.subtract(n, ca))/np.pi))
            # print()

            ######### Get Distances to surrounding atoms
            # print('Distance of point_1 to C: {}'.format(np.linalg.norm(np.subtract(point_1, c))))
            # print('Distance of point_1 to O: {}'.format(np.linalg.norm(np.subtract(point_1, o))))
            # print('Distance of point_2 to C: {}'.format(np.linalg.norm(np.subtract(point_2, c))))
            # print('Distance of point_2 to O: {}'.format(np.linalg.norm(np.subtract(point_2, o))))
            # print()

            # Only assume coordinates to be valid if the distance to the c is bigger than 1 and if the distance to the
            # o is bigger than 1
            if not np.linalg.norm(np.subtract(point_1, c)) < 1 and not np.linalg.norm(np.subtract(point_1, o)) < 1:
                h_atoms.append(point_1)

            if not np.linalg.norm(np.subtract(point_2, c)) < 1 and not np.linalg.norm(np.subtract(point_2, o)) < 1:
                h_atoms.append(point_2)

        else:
            point_1 = support_point + solutions[0] * direction_vec

            ######### Check if point lies in planes
            # print(np.dot(plane_1[:3], point_1) - plane_1[3])
            # print(np.dot(plane_2[:3], point_1) - plane_2[3])
            # print()

            ######### Check Distance to N atom -> should be ~0.97A
            # print('Distance to N point_1: {}'.format(np.linalg.norm(np.subtract(n, point_1))))
            # print()

            ######### Check if angle == ~119°
            # print(180-(180*angle(np.subtract(point_1, n), np.subtract(n, ca))/np.pi))
            # print()

            ######### Get Distances to surrounding atoms
            # print('Distance to C: {}'.format(np.linalg.norm(np.subtract(point_1,c))))
            # print('Distance to O: {}'.format(np.linalg.norm(np.subtract(point_1,o))))
            # print()

            # Only assume coordinates to be valid if the distance to the c is bigger than 1 and if the distance to the
            # o is bigger than 1
            if not np.linalg.norm(np.subtract(point_1, c)) < 1 and not np.linalg.norm(np.subtract(point_1, o)) < 1:
                h_atoms.append(point_1)

    return h_atoms


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



def main():
    file = 'supplementary/1b35.pdb'
    pdb_parser = PDBParser(QUIET=True)
    name = os.path.basename(file).split('.')[0]
    structure = pdb_parser.get_structure(name, file)

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in AMINO_ACIDS:
                    residues.append(residue)
        break
    
    # save the H coordinates in list
    h_coord_per_res = []


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

        #print([phi_angle, psi_angle, calculate_h(np.array(o_coord), np.array(c_coord), np.array(c_alpha_coord),
         #                                       np.array(n_coord))])

        h_coord_per_res.append(calculate_h(np.array(o_coord), np.array(c_coord), np.array(c_alpha_coord),
                                                np.array(n_coord)))
         
        #break
    dssp.dssp(residues, h_coord_per_res) 
        


if __name__ == '__main__':
    main()




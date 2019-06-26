'''
This file estimates the H-atom position for the given information
    - 0.97 Angstrom N-H bond length
    - 119 degrees angle of the C_alpha-N-H bonds
    - planarity of the O=CN-H system
'''

from Bio.PDB import *
import os
import numpy as np


AMINO_ACIDS = ['ALA', 'GLY', 'PHE', 'ILE', 'MET', 'LEU', 'PRO', 'VAL', 'ASP', 'GLU', 'LYS', 'ARG', 'SER', 'THR', 'TYR',
               'HIS', 'CYS', 'ASN', 'GLN', 'TRP']

ATOMS = ['C', 'N', 'CA', 'O']


# This function solves the standard quadratic formula, since th problem can be reduced to that
def solve_quadratic(a, b, c):
    x_1 = (-b + np.sqrt(b**2 - 4 * a * c))/(2*a)
    x_2 = (-b - np.sqrt(b**2 - 4 * a * c))/(2*a)
    return [x_1,x_2]


def calculate_h(o, c, ca, n):
    # distance of the N-H bond as given by the task description
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

    # n_prime is support vector to plane
    n_prime = np.add(ca, normal_2)

    plane_2 = np.append(normal_2,np.dot(n_prime,normal_2))

    ###################################################################################
    # solve linear system
    system = np.array([normal_1, normal_2, np.cross(normal_1,normal_2)])
    solution = np.array([np.dot(normal_1,o),np.dot(n_prime,normal_2),0])

    # get point on line -> defines line with direction vector
    support_point = np.linalg.solve(system,solution)

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

    h_atoms = []
    if solutions is not None:
        # if there are two solutions for the quadratic formula consider both as possible coordinates
        if len(solutions) == 2:
            # Calculate actual §d point
            point_1 = support_point + solutions[0] * direction_vec
            point_2 = support_point + solutions[1] * direction_vec

            # Only assume coordinates to be valid if the distance to the c is bigger than 1 and if the distance to the
            # o is bigger than 1
            if not np.linalg.norm(np.subtract(point_1, c)) < 1 and not np.linalg.norm(np.subtract(point_1, o)) < 1:
                h_atoms.append(point_1)

            if not np.linalg.norm(np.subtract(point_2, c)) < 1 and not np.linalg.norm(np.subtract(point_2, o)) < 1:
                h_atoms.append(point_2)

        else:
            # Calculate actual 3d point
            point_1 = support_point + solutions[0] * direction_vec

            # Only assume coordinates to be valid if the distance to the c is bigger than 1 and if the distance to the
            # o is bigger than 1
            if not np.linalg.norm(np.subtract(point_1, c)) < 1 and not np.linalg.norm(np.subtract(point_1, o)) < 1:
                h_atoms.append(point_1)

    # The list either contains both possible points or only one point if the other one can be discarded due to physical
    # imposibilities
    return h_atoms
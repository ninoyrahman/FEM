# FEM 2D Heat equation solver
# Fixed temperature at two sides

# import
import os
import sys
sys.path.append(os. getcwd())
sys.path.append(os. getcwd() + '\\data')

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from FEM_heat2d import FEheat2D
from FEM_mesh import Mesh

if __name__ == '__main__':
    """
    Problem setup
    """
    # Right-hand side function
    def RHS(_x, _y):
        return 0.0

    # Domain space
    iteration_max = 100
    x_min, x_max, n_x = -0.5, 0.5, 50
    y_min, y_max, n_y = -0.5, 0.5, 50

    """ 
    Create a mesh
    """
    print('Create a mesh')
    # Create a mesh object and a boundary condition dictionary
    mesh = Mesh(x_min, x_max, n_x, y_min, y_max, n_y)
    # Mesh data
    inner_points = [i for i in range(len(mesh.tri.points))
                    if i not in mesh.boundary_points]
    print("The mesh has {} boundary nodes, and {} internal nodes".format(
        len(mesh.boundary_points), len(inner_points)))

    """ 
    Set boundary conditions
    """
    print('Set boundary conditions')
    # Set the Dirichlet boundary conditions
    for point_index in mesh.boundary_points:
        p = mesh.tri.points[point_index]
        if p[0] == x_min:  # West
            mesh.bc_points["dirichlet"][point_index] = 0.0
        if p[0] == x_max:  # East
            mesh.bc_points["dirichlet"][point_index] = 1.0

    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.tri.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[1] == y_min and p2[1] == y_min:  # South
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_max and p2[1] == y_max:  # North
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]

    """ 
    Solve the 2D heat problem
    """
    print('Solve the 2D heat problem')
    # Set a simulation object
    heat = FEheat2D(mesh, RHS, _gpu=True, _sparse=True)
    
    # Initialize
    heat.initialze()
    
    # Solve
    print('iteration=')
    for i in range(iteration_max):
        heat.solve()
        print(i, end=' ', flush=True)
    
    # Get the result
    u = heat.u

    if heat.gpu:
        # Inspect pool usage
        print('')
        print(f"Used: {heat.mp.used_bytes() / 1e6:.2f} MB")
        print(f"Total allocated: {heat.mp.total_bytes() / 1e6:.2f} MB")
            
        # Free unused blocks back to OS
        heat.mp.free_all_blocks()

    """ 
    Plot
    """
    print('Plot')
    from FEM_plot import plot3d
    plt.rcParams['figure.figsize'] = 6, 6
    
    # Analytical solution
    x = mesh.tri.points[:, 0]
    y = mesh.tri.points[:, 1]
    u_exact = x
    
    u_exact_tmp = u_exact.reshape((n_x, n_y))
    u_tmp = u.reshape((n_x, n_y))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot3d(ax, u.flatten(), varmin=0, varmax=1, x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1])
    
    plt.tight_layout()
    plt.savefig('heat_2D_ex001.png')
    plt.close()
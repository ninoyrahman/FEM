# FEM 2D Heat equation solver
# Gaussian pulse initial condition

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
from FEM_plot import plot3d

if __name__ == '__main__':
    """
    Problem setup
    """
    # Right-hand side function
    def RHS(_x, _y):
        if _x**2 + _y**2 < 0.1:
            return 0.0
        else:
            return 0.0

    # Domain space
    iteration_max = 10
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

    x = mesh.tri.points[:, 0]
    y = mesh.tri.points[:, 1]
    n_points = mesh.tri.npoints
    
    """ 
    Set boundary conditions 
    """
    print('Set boundary conditions')
    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.tri.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # West
            mesh.bc_points["neumann_edge"][ch_idx] = [0.0, 0.0]
        elif p1[0] == x_max and p2[0] == x_max:  # East
            mesh.bc_points["neumann_edge"][ch_idx] = [0.0, 0.0]
        if p1[1] == y_min and p2[1] == y_min:  # South
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_max and p2[1] == y_max:  # North
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]

    """ 
    Solve the 2D heat problem
    """
    print('Solve the 2D heat problem')
    u0 = np.zeros((n_points, 1))
    u0[:, 0] = np.exp(-(x**2 + y**2)/(2.0 * 0.15**2))

    # Set a simulation object
    heat = FEheat2D(_mesh=mesh, _f=RHS, _u=u0, _gpu=True, _sparse=True)

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
    plt.rcParams['figure.figsize'] = 6, 6
    
    u_tmp = u.reshape((n_x, n_y))
    u0_tmp = u0.reshape((n_x, n_y))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot3d(ax, u0.flatten(), varmin=0, varmax=1, x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1], label=r'$u_0$', cmap=cm.berlin)
    plot3d(ax, u.flatten(), varmin=0, varmax=1, x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1], label=r'$u$')
    
    plt.tight_layout()
    plt.savefig('heat_2D_ex002.png')
    plt.close()
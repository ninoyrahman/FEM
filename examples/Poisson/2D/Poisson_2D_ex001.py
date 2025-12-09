# FEM 2D Poisson equation solver
# Sinusoidal source term

# import
import os
import sys
sys.path.append(os. getcwd())
sys.path.append(os. getcwd() + '/data/')

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from FEM_poisson2d import FEPoisson2D
from FEM_mesh import Mesh
from FEM_plot import plot3d

if __name__ == '__main__':
    """
    Problem setup
    """
    # Right-hand side function
    def RHS(_x, _y):
        return 2 * np.sin(_x) * np.sin(_y) 

    # Domain space
    x_min, x_max, n_x = 0, np.pi/2, 100
    y_min, y_max, n_y = 0, np.pi/2, 100

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
    u00 = 0
    for point_index in mesh.boundary_points:
        p = mesh.tri.points[point_index]
        if p[0] == 0 and p[1] == 0:
            mesh.bc_points["dirichlet"][point_index] = u00

    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.tri.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == 0 and p2[0] == 0:  # x = 0, West
            mesh.bc_points["neumann_edge"][ch_idx] = [-np.sin(p1[1]), -np.sin(p2[1])]
        elif p1[0] == np.pi/2 and p2[0] == np.pi/2:  # x = 1, East
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == 0 and p2[1] == 0:  # y = 0, South
            mesh.bc_points["neumann_edge"][ch_idx] = [-np.sin(p1[0]), -np.sin(p2[0])]
        elif p1[1] == np.pi/2 and p2[1] == np.pi/2:  # y = 1, North
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]

    """ 
    Solve the 2D Poisson problem 
    """
    print('Solve the 2D Poisson problem')
    # Set a simulation object
    poisson = FEPoisson2D(mesh, RHS, _gpu=True, _sparse=True)
    
    # Initialize
    poisson.process()
    
    # Solve
    poisson.solve()
    
    # Get the result
    u = poisson.u

    if poisson.gpu:
        # Inspect pool usage
        print('')
        print(f"Used: {poisson.mp.used_bytes() / 1e6:.2f} MB")
        print(f"Total allocated: {poisson.mp.total_bytes() / 1e6:.2f} MB")
            
        # Free unused blocks back to OS
        poisson.mp.free_all_blocks()

    """ 
    Plot
    """
    print('Plot')
    # Analytical solution
    x = mesh.tri.points[:, 0]
    y = mesh.tri.points[:, 1]
    u_exact = np.sin(x) * np.sin(y) + u00

    plt.rcParams['figure.figsize'] = 12, 6
    
    u_exact_tmp = u_exact.reshape((n_x, n_y))
    u_tmp = u.reshape((n_x, n_y))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot3d(ax, u.flatten(), varmin=0, varmax=1, x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1])
    ax.scatter(x, y, u_exact, marker='o', c='black', s=1, label="Analytical")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(y[:n_y], u_tmp)
    plt.plot(y[:n_y], u_exact_tmp, color='k', ls='--', lw=1)
    plt.ylim([0, 1.01])
    plt.xlabel(r'$y$')
    plt.ylabel(r'$u(x,y)$')
    
    plt.tight_layout()
    output_path = os.getcwd() + '/output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig('output/Poisson_2D_ex001.png')
    plt.close()
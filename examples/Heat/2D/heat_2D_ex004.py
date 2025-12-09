# FEM 2D Heat equation solver
# Fixed temperature airfoil in the middle

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

from FEM_heat2d_cavity import FEheat2D
from FEM_mesh import Mesh_cavity_outline
from FEM_plot import plot3d

if __name__ == '__main__':
    """
    Problem setup
    """
    # Right-hand side function
    def RHS(_x, _y):
        return 0.0

    """
    Domain space
    """
    iteration_max = 100
    x_min, x_max, n_x = -0.5, 1.5, 100
    y_min, y_max, n_y = -0.3, 0.3, 150

    """ 
    Create a mesh
    """
    print('Create a mesh')
    # Create a mesh object and a boundary condition dictionary
    mesh = Mesh_cavity_outline(x_min, x_max, n_x, y_min, y_max, n_y, outline=np.loadtxt('data/naca6412.dat'))
    # Mesh data
    inner_points = [i for i in range(len(mesh.tri.points))
                    if i not in mesh.boundary_points]
    print("The mesh has {} boundary nodes, and {} internal nodes".format(
        len(mesh.boundary_points), len(inner_points)))

    x = mesh.tri.points[:, 0]
    y = mesh.tri.points[:, 1]
    n_points = mesh.npoints

    """ 
    Set boundary conditions
    """
    print('Set boundary conditions')
    # Set the Dirichlet boundary conditions
    for point_index in mesh.boundary_points:
        if mesh.cbflg[point_index]:  # cavity boundary
            mesh.bc_points["dirichlet"][point_index] = 1.0

    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.tri.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # West
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[0] == x_max and p2[0] == x_max:  # East
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_min and p2[1] == y_min:  # South
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_max and p2[1] == y_max:  # North
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]

    """ 
    Solve the 2D heat problem
    """
    print('Solve the 2D heat problem')
    u0 = np.zeros((n_points, 1))

    # Set a simulation object
    heat = FEheat2D(_mesh=mesh, _f=RHS, _u=u0, _gpu=True, _sparse=True)

    # Initialize
    heat.initialze()

    # Solve
    print('iteration=')
    for i in range(iteration_max):
        print(i, end=' ')
        
        heat.solve()
    
        # Get the result
        u = np.array([[heat.u[mesh.pmap[i]] if mesh.pmap[i] > -1 else np.ones(1)] for i in range(mesh.tri.npoints)]).reshape(mesh.tri.npoints)

    # Inspect pool usage
    if heat.gpu:
        print('')
        print(f"Used: {heat.mp.used_bytes() / 1e6:.2f} MB")
        print(f"Total allocated: {heat.mp.total_bytes() / 1e6:.2f} MB")
                
        # Free unused blocks back to OS
        heat.mp.free_all_blocks()

    """
    Plot
    """
    print('Plot')
    mpl.rcParams['figure.figsize'] = 6, 6
    
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 1, 1)
    surf = ax.tricontourf(x, y, u.flatten(), cmap=cm.coolwarm)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    cbar = plt.colorbar(surf, ax=ax, orientation="vertical", pad=0.02, ticks=np.linspace(0, 1, 11))
    cbar.ax.set_ylabel(r'$u$')
    
    plt.tight_layout()
    output_path = os.getcwd() + '/output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig('output/heat_2D_ex004.png')
    plt.close()
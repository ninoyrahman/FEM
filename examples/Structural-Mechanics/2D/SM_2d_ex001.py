# FEM 2D structural mechanics equation solver
# Trapezoidal beam with traction

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

from FEM_SM import FESM2D
from FEM_mesh import Mesh_from_FreeCAD
from trapezoid import faces

if __name__ == '__main__':
    """
    Problem setup
    """
    # Right-hand side function
    def RHS(_x, _y):
        return np.array([0., - 1.2e3 * 9.81]) # gravity in N/m3

    # setup
    iteration_max = 1
    T = 2e5 # surface force in N/m

    """ 
    Create a mesh
    """
    print('Create a mesh')
    # Create a mesh object and boundary condition dictionaries
    simplices_coord = np.round(np.array(faces, dtype=np.float32)[:, :2], decimals=3) / 100.0

    mesh = Mesh_from_FreeCAD(_faces=simplices_coord, ratio=0.01)
    # Mesh data
    inner_points = [i for i in range(len(mesh.tri.points))
                    if i not in mesh.boundary_points]
    print("The mesh has {} boundary nodes, and {} internal nodes".format(
        len(mesh.boundary_points), len(inner_points)))

    x = mesh.tri.points[:, 0]
    y = mesh.tri.points[:, 1]
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    
    """ 
    Set boundary conditions
    """
    print('Set boundary conditions')
    # Set the Dirichlet boundary conditions
    for point_index in mesh.boundary_points:
        p = mesh.tri.points[point_index]
        if p[0] == x_min:  # West
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
            
    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.tri.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # West
            pass
        elif p1[1] == y_min and p2[1] == y_min:  # South
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_max and p2[1] == y_max:  # North
            mesh.bc_points_u["neumann_edge"][ch_idx] = [T, T]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        else:
            mesh.bc_points_u["neumann_edge"][ch_idx] = [T/np.sqrt(2.0), T/np.sqrt(2.0)] # T/np.sqrt(2.0)
            mesh.bc_points_v["neumann_edge"][ch_idx] = [-T/np.sqrt(2.0), -T/np.sqrt(2.0)]


    """ 
    Solve the 2D SM problem
    """
    print('Solve the 2D SM problem')
    # Set a simulation object
    sm = FESM2D(mesh, _f=RHS, _nu=0.499, _E=0.1e9, _rho=1.2e3, _plain_stress=True, _gpu=True, _sparse=False, _dt=0.001) # rubber

    # Initialize
    sm.initialze()

    print('iteration=')
    for i in range(iteration_max):
        sm.solve()
        print(i, end=' ', flush=True)
    
    # Get the result
    u = sm.u

    if sm.gpu:
        # Inspect pool usage
        print('')
        print(f"Used: {sm.mp.used_bytes() / 1e6:.2f} MB")
        print(f"Total allocated: {sm.mp.total_bytes() / 1e6:.2f} MB")
            
        # Free unused blocks back to OS
        sm.mp.free_all_blocks()

    """ 
    Plot
    """
    print('Plot')
    plt.rcParams['figure.figsize'] = 12, 6
    
    # Analytical solution
    x = mesh.tri.points[:, 0]
    y = mesh.tri.points[:, 1]
    
    ux = u[0::2, 0]
    uy = u[1::2, 0]
    u_tot = np.sqrt(ux**2+uy**2)
    ux_norm = ux / u_tot
    uy_norm = uy / u_tot
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    surf = ax.tricontourf(x, y, u_tot, cmap=cm.coolwarm)
    ax.quiver(x[::5], y[::5], ux_norm[::5], uy_norm[::5], units='xy', scale=20, alpha=0.5)
    ax.scatter((x+ux)[mesh.tri.boundary_points], (y+uy)[mesh.tri.boundary_points], color='k', s=10)
    plt.gca().set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar = plt.colorbar(surf, ax=ax, orientation="vertical", pad=0.02, ticks=np.linspace(u_tot.min(), u_tot.max(), 11))
    cbar.ax.set_ylabel(r'$u$')
    
    ax = fig.add_subplot(1, 2, 2)
    plt.triplot(x, y, mesh.tri.simplices, alpha=0.5)
    plt.scatter(x, y, color='r', s=1)
    plt.scatter(x[mesh.tri.boundary_points], y[mesh.tri.boundary_points], color='k', s=10)
    plt.gca().set_aspect('equal')
    plt.title('triangles')
    
    plt.tight_layout()
    plt.savefig('SM_2D_ex001.png')
    plt.close()
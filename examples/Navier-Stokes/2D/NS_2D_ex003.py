# FEM 2D NS equations solver
# Flow over an airfoil inside a channel

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

from FEM_NS_cavity import FENS2D
from FEM_mesh import Mesh_from_FreeCAD_with_cavity_outline_ns
from square_hd_2d import faces
from FEM_plot import plot3d

if __name__ == '__main__':
    """
    Problem setup
    """
    # Right-hand side function
    def RHS(_x, _y):
        return 0.0

    # Domain space
    iteration_max = 50
    plot = True

    """ 
    Create a mesh
    """
    print('Create a mesh')
    # Create a mesh object and a boundary condition dictionary
    simplices_coord = np.round(np.array(faces, dtype=np.float32)[:, :2])
    simplices_coord[:, 0] /= 50
    simplices_coord[:, 1] *= (1/200) * (3/2.5)
    simplices_coord[:, 0] -= 0.5
    simplices_coord[:, 1] -= 0.25 * (3/2.5)
    
    mesh = Mesh_from_FreeCAD_with_cavity_outline_ns(_faces=simplices_coord, outline=np.loadtxt('data/naca6412.dat'), ratio=1)
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
            mesh.bc_points_u["dirichlet"][point_index] = 1.0
        if p[1] == y_min:  # South
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
        if p[1] == y_max:  # North
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
        if mesh.cbflg[point_index]:  # cavity boundary
            mesh.bc_points_u["dirichlet"][point_index] = 0.0
            mesh.bc_points_v["dirichlet"][point_index] = 0.0
            
    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.tri.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # West
            mesh.bc_points_p["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        if p1[0] == x_max and p2[0] == x_max:  # East
            mesh.bc_points_p["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        if p1[1] == y_min and p2[1] == y_min:  # South
            mesh.bc_points_p["neumann_edge"][ch_idx] = [0, 0]
        if p1[1] == y_max and p2[1] == y_max:  # North
            mesh.bc_points_p["neumann_edge"][ch_idx] = [0, 0]
        if mesh.cbflg[p1_idx] and mesh.cbflg[p2_idx]:  # cavity boundary
            mesh.bc_points_p["neumann_edge"][ch_idx] = [0, 0]

    """ 
    Solve the 2D NS problem
    """
    print('Solve the 2D NS problem')
    # Set a simulation object
    ns = FENS2D(mesh, RHS, _gpu=True, _sparse=True, _dt=0.001)
    
    # Initialize
    ns.initialze()

    # Solve
    print('iteration=')
    for i in range(iteration_max):
        ns.solve()
        print(i, end=' ', flush=True)
    
        # Get the result
        u = np.array([[ns.u[mesh.pmap[i]] if mesh.pmap[i] > -1 else np.zeros(1)] for i in range(mesh.tri.npoints)]).reshape(mesh.tri.npoints).flatten()
        v = np.array([[ns.v[mesh.pmap[i]] if mesh.pmap[i] > -1 else np.zeros(1)] for i in range(mesh.tri.npoints)]).reshape(mesh.tri.npoints).flatten()
        p = np.array([[ns.p[mesh.pmap[i]] if mesh.pmap[i] > -1 else 0.5*np.ones(1)] for i in range(mesh.tri.npoints)]).reshape(mesh.tri.npoints).flatten()

    if ns.gpu:
        # Inspect pool usage
        print('')
        print(f"Used: {ns.mp.used_bytes() / 1e6:.2f} MB")
        print(f"Total allocated: {ns.mp.total_bytes() / 1e6:.2f} MB")
            
        # Free unused blocks back to OS
        ns.mp.free_all_blocks()

    """ 
    Plot
    """
    print('Plot')
    plt.rcParams['figure.figsize'] = 12, 12
    
    v2 = u**2 + v**2
    u_norm = u / np.sqrt(v2)
    v_norm = v / np.sqrt(v2)
    
    fig = plt.figure()
    ax = fig.add_subplot(3, 2, 1, projection='3d')
    plot3d(ax, p, varmin=p.min(), varmax=p.max(), x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1], label=r'$P$')
    
    ax = fig.add_subplot(3, 2, 2)
    surf = ax.tricontourf(x, y, p, cmap=cm.coolwarm)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=15, alpha=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar = plt.colorbar(surf, ax=ax, orientation="vertical", pad=0.02, ticks=np.linspace(p.min(), p.max(), 11))
    cbar.ax.set_ylabel(r'$p$')
    
    ax = fig.add_subplot(3, 2, 3, projection='3d')
    plot3d(ax, u, varmin=u.min(), varmax=u.max(), x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1], label=r'$u$')
    
    ax = fig.add_subplot(3, 2, 4)
    surf = ax.tricontourf(x, y, u, cmap=cm.coolwarm)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=15, alpha=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar = plt.colorbar(surf, ax=ax, orientation="vertical", pad=0.02, ticks=np.linspace(u.min(), u.max(), 11))
    cbar.ax.set_ylabel(r'$u$')
    
    ax = fig.add_subplot(3, 2, 5, projection='3d')
    plot3d(ax, v2, varmin=v2.min(), varmax=v2.max(), x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1], label=r'$v^2$', azim=45)
    
    ax = fig.add_subplot(3, 2, 6)
    surf = ax.tricontourf(x, y, v2, cmap=cm.coolwarm)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=15, alpha=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar = plt.colorbar(surf, ax=ax, orientation="vertical", pad=0.02, ticks=np.linspace(v2.min(), v2.max(), 11))
    cbar.ax.set_ylabel(r'$v^2$')
    
    plt.tight_layout()
    plt.savefig('NS_2D_ex003.png')
    plt.close()
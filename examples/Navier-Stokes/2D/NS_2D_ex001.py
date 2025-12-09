# FEM 2D NS equations solver
# Flow through a nozzle

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

from FEM_NS import FENS2D
from FEM_mesh import Mesh_from_FreeCAD_ns
from nozzle_2d import faces
from FEM_plot import plot3d

if __name__ == '__main__':
    """
    Problem setup
    """
    # Right-hand side function
    def RHS(_x, _y):
        return 0.0

    iteration_max = 50

    """ 
    Create a mesh
    """
    print('Create a mesh')
    # Create a mesh object and a boundary condition dictionary
    simplices_coord= np.round(np.array(faces, dtype=np.float32)[:, :2])
    simplices_coord[:, 1] -= 25
    simplices_coord /= 100
    mesh = Mesh_from_FreeCAD_ns(_faces=simplices_coord, ratio=0.04)
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
        if p[0] == x_min:  # x = x_min, West
            # mesh.bc_points_u["dirichlet"][point_index] = 1.0
            mesh.bc_points_p["dirichlet"][point_index] = 1.0
        if p[0] == x_max:  # x = x_max, East
            mesh.bc_points_p["dirichlet"][point_index] = 0.5
        if (p[0] != x_min and p[0] != x_max) and p[1] < 0:  # South
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
        if (p[0] != x_min and p[0] != x_max) and p[1] > 0:  # North
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
            
    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.tri.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # x = x_min, West
            # mesh.bc_points_p["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        if p1[0] == x_max and p2[0] == x_max:  # x = x_max, East
            # mesh.bc_points_p["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        if (p1[0] != x_min and p1[0] != x_max and p1[1] < 0) and (p2[0] != x_min and p2[0] != x_max and p2[1] < 0):  # South
            # print('South', p1_idx, p2_idx, p1, p2)
            mesh.bc_points_p["neumann_edge"][ch_idx] = [0, 0]
        if (p1[0] != x_min and p1[0] != x_max and p1[1] > 0) and (p2[0] != x_min and p2[0] != x_max and p2[1] > 0):  #  North
            # print('North', p1_idx, p2_idx, p1, p2)
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
    u = ns.u
    v = ns.v
    p = ns.p

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
    
    u_tot = np.sqrt(u**2 + v**2)
    mask = u_tot > 0
    u_norm = np.zeros_like(u, dtype=np.float32)
    v_norm = np.zeros_like(v, dtype=np.float32)
    u_norm[mask] = u[mask] / u_tot[mask]
    v_norm[mask] = v[mask] / u_tot[mask]
    
    fig = plt.figure()
    ax = fig.add_subplot(3, 2, 1, projection='3d')
    plot3d(ax, p.flatten(), varmin=p.flatten().min(), varmax=p.flatten().max(), x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1], label=r'$P$', azim=-30)
    
    ax = fig.add_subplot(3, 2, 2)
    surf = ax.tricontourf(x, y, p.flatten(), cmap=cm.coolwarm)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=10, alpha=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar = plt.colorbar(surf, ax=ax, orientation="vertical", pad=0.02, ticks=np.linspace(p.flatten().min(), p.flatten().max(), 11))
    cbar.ax.set_ylabel(r'$p$')
    
    ax = fig.add_subplot(3, 2, 3, projection='3d')
    plot3d(ax, u.flatten(), varmin=u.flatten().min(), varmax=u.flatten().max(), x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1], label=r'$u$', azim=-30)
    
    ax = fig.add_subplot(3, 2, 4)
    surf = ax.tricontourf(x, y, u.flatten(), cmap=cm.coolwarm)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=10, alpha=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar = plt.colorbar(surf, ax=ax, orientation="vertical", pad=0.02, ticks=np.linspace(u.flatten().min(), u.flatten().max(), 11))
    cbar.ax.set_ylabel(r'$u$')
    
    ax = fig.add_subplot(3, 2, 5, projection='3d')
    plot3d(ax, v.flatten(), varmin=v.flatten().min(), varmax=v.flatten().max(), x=mesh.tri.points[:, 0], y=mesh.tri.points[:, 1], label=r'$v$', azim=-30)
    
    ax = fig.add_subplot(3, 2, 6)
    surf = ax.tricontourf(x, y, v.flatten(), cmap=cm.coolwarm)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=10, alpha=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    cbar = plt.colorbar(surf, ax=ax, orientation="vertical", pad=0.02, ticks=np.linspace(v.flatten().min(), v.flatten().max(), 11))
    cbar.ax.set_ylabel(r'$v$')
    
    plt.tight_layout()
    output_path = os.getcwd() + '/output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig('output/NS_2D_ex001.png')
    plt.close()
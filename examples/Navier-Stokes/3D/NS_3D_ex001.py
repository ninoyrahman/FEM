# FEM 3D NS equations solver
# Flow through a nozzle

# import
import os
import sys
sys.path.append(os. getcwd() + '\\3D')
sys.path.append(os. getcwd() + '\\data')

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from FEM_NS_3d import FENS3D
from FEM_mesh_3d import Mesh_from_FreeCAD
from nozzle_3d import create_nodes, create_elements

class FemMesh:
    def __init__(self):
        self.nodes = dict()
        self.faces = []

    def addNode(self, x, y, z, elm):
        self.nodes[elm] = [x, y, z]

    def addVolume(self, nlist, elm):
        pidx1, pidx2, pidx3, pidx4 = nlist
        self.faces.append(self.nodes[pidx1])
        self.faces.append(self.nodes[pidx2])
        self.faces.append(self.nodes[pidx3])
        self.faces.append(self.nodes[pidx4])

if __name__ == '__main__':
    """
    Problem setup
    """
    # Right-hand side function
    def RHS(_x, _y, _z):
        return 0.0

    iteration_max = 50

    """ 
    Create a mesh
    """
    print('Create a mesh')
    # Create a mesh object and a boundary condition dictionary
    femmesh = FemMesh()
    exitCode = create_nodes(femmesh)
    exitCode = create_elements(femmesh)
    simplices_coord = np.round(np.array(femmesh.faces, dtype=np.float32)[:, :], 3) / 100
    mesh = Mesh_from_FreeCAD(_faces=simplices_coord, alpha=34)
    # Mesh data
    inner_points = [i for i in range(len(mesh.tri.points))
                    if i not in mesh.boundary_points]
    print("The mesh has {} boundary nodes, and {} internal nodes".format(
        len(mesh.boundary_points), len(inner_points)))

    x = mesh.tri.points[:, 0]
    y = mesh.tri.points[:, 1]
    z = mesh.tri.points[:, 2]
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    z_min = z.min()
    z_max = z.max()

    """ 
    Set boundary conditions
    """
    print('Set boundary conditions')
    # Set the Dirichlet boundary conditions
    for point_index in mesh.boundary_points:
        p = mesh.tri.points[point_index]
        if p[0] == x_min:  # West
            mesh.bc_points["dirichlet"][point_index] = 1.0
        elif p[0] == x_max:  # East
            mesh.bc_points["dirichlet"][point_index] = 0.5
        else:
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
            
    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # West
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        elif p1[0] == x_max and p2[0] == x_max:  # East
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        else:
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]

    """ 
    Solve the 3D NS problem
    """
    print('Solve the 3D NS problem')
    # Set a simulation object
    ns = FENS3D(mesh, RHS, _gpu=True, _sparse=True, _dt=0.001)

    # Initialize
    ns.initialize()

    # Solve
    print('iteration=')
    for i in range(iteration_max):
        ns.solve()
        print(i, end=' ', flush=True)
    
    # Get the result
    u = ns.u.flatten()
    v = ns.v.flatten()
    w = ns.w.flatten()
    p = ns.p.flatten()

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
    u_tot = np.sqrt(u**2 + v**2 + w**2)
    u_norm = u / u_tot
    v_norm = v / u_tot
    w_norm = w / u_tot
    
    grid_x, grid_y, grid_z = np.mgrid[x.min():x.max():100j, y.min():y.max():100j, z.min():z.max():100j]
    u_intp = sp.interpolate.griddata(mesh.tri.points, u_tot, (grid_x, grid_y, grid_z), method='linear')
    lrange = np.linspace(u_tot.min(), u_tot.max(), 6)
    label = r'$u_{tot}$'
    cmap = cm.coolwarm
    
    plt.rcParams['figure.figsize'] = 16, 6
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    surf = ax.contourf(grid_x[:, :, 50], grid_y[:, :, 50], u_intp[:, :, 50], cmap=cmap)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=10, alpha=0.5)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')
    
    ax = fig.add_subplot(1, 3, 2)
    surf = ax.contourf(grid_x[:, 50, :], grid_z[:, 50, :], u_intp[:, 50, :], cmap=cmap)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([z.min(), z.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_aspect('equal')
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')
    
    ax = fig.add_subplot(1, 3, 3)
    surf = ax.contourf(grid_y[50, :, :], grid_z[50, :, :], u_intp[50, :, :], cmap=cmap)
    ax.set_xlim([y.min(), y.max()])
    ax.set_ylim([z.min(), z.max()])
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$z$')
    ax.set_aspect('equal')
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')
    
    plt.tight_layout()
    plt.savefig('NS_3D_ex001.png')
    plt.close()
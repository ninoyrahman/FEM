# FEM 3D Heat equation solver
# Heating around airfoil

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

from rectangular_prism_3d import create_nodes, create_elements
from FEM_mesh_3d import Mesh_from_FreeCAD
from FEM_heat_3d import FEheat3D

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
        return np.array([0., 0., 0.])

    """ 
    Create a rectangular mesh
    """
    print('Create a mesh')
    # Create a mesh object with Delaunay mesh and a boundary condition dictionary
    femmesh = FemMesh()
    exitCode = create_nodes(femmesh)
    exitCode = create_elements(femmesh)
    simplices_coord = np.round(np.array(femmesh.faces, dtype=np.float32)[:, :], 2) / 100
    mesh = Mesh_from_FreeCAD(_faces=simplices_coord, alpha=34, outline=np.loadtxt('naca6412.dat')) #, outline=np.loadtxt('naca6412.dat')
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
        if mesh.cbflg[point_index]:  # cavity boundary
            mesh.bc_points["dirichlet"][point_index] = 1.0
            
    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # West
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        if p1[0] == x_max and p2[0] == x_max:  # East
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[2] == z_min and p2[2] == z_min:  # Bottom
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[2] == z_max and p2[2] == z_max:  # Top
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_min and p2[1] == y_min:  # South
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_max and p2[1] == y_max:  # North
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]

    """ 
    Solve the 3D heat transfer problem
    """
    print('Solve the 3D heat transfer problem')
    # Set a simulation object
    heat = FEheat3D(mesh, _f=RHS, _gpu=True, _sparse=True, _dt=1e-3)

    # Initialize
    heat.initialze()

    iteration_max = 500
    print('iteration=')
    for i in range(iteration_max):
        heat.solve()
        print(i, end=' ', flush=True)
    
    # Get the result
    u = np.array([[heat.u[mesh.pmap[i]] if mesh.pmap[i] > -1 else np.ones(1)] for i in range(mesh.tri.npoints)]).reshape(mesh.tri.npoints)

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
    grid_x, grid_y, grid_z = np.mgrid[x.min():x.max():100j, y.min():y.max():100j, z.min():z.max():100j]
    u_intp = sp.interpolate.griddata(mesh.tri.points, u, (grid_x, grid_y, grid_z), method='linear')
    lrange = np.linspace(u_intp.min(), u_intp.max(), 2)
    label = r'$u$'
    cmap = cm.coolwarm
    
    xidx = 50
    yidx = 60
    zidx = 50
    aspect='equal' # 'auto'
    
    plt.rcParams['figure.figsize'] = 16, 6
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    surf = ax.contourf(grid_x[:, :, zidx], grid_y[:, :, zidx], u_intp[:, :, zidx], cmap=cmap)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect(aspect)
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')
    
    ax = fig.add_subplot(1, 3, 2)
    surf = ax.contourf(grid_x[:, yidx, :], grid_z[:, yidx, :], u_intp[:, yidx, :], cmap=cmap)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([z.min(), z.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_aspect(aspect)
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')
    
    ax = fig.add_subplot(1, 3, 3)
    surf = ax.contourf(grid_z[xidx, :, :], grid_y[xidx, :, :], u_intp[xidx, :, :], cmap=cmap)
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlim([z.min(), z.max()])
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect(aspect)
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')
    
    plt.tight_layout()
    plt.savefig('heat_3D_ex002.png')
    plt.close()
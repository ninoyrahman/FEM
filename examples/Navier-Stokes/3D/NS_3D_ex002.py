# FEM 3D NS equations solver
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

from FEM_NS_3d import FENS3D
from FEM_mesh_3d import Mesh_from_FreeCAD
from rectangular_prism_3d import create_nodes, create_elements

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
    simplices_coord = np.round(np.array(femmesh.faces, dtype=np.float32)[:, :], 2) / 100
    mesh = Mesh_from_FreeCAD(_faces=simplices_coord, alpha=34, outline=np.loadtxt('data/naca6412.dat'))
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
            mesh.bc_points_u["dirichlet"][point_index] = 1.0
        elif p[1] == y_min:  # South
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
        elif p[1] == y_max:  # North
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
        elif p[2] == z_min:  # Bottom
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
        elif p[2] == z_min:  # Top
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
        if mesh.cbflg[point_index]:  # cavity boundary
            mesh.bc_points_u["dirichlet"][point_index] = 0.0
            mesh.bc_points_v["dirichlet"][point_index] = 0.0
            
    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # West
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        elif p1[0] == x_max and p2[0] == x_max:  # East
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_min and p2[1] == y_min:  # South
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_max and p2[1] == y_max:  # North
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[2] == z_min and p2[2] == z_min:  # Bottom
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        elif p1[2] == z_max and p2[2] == z_max:  # Top
            mesh.bc_points["neumann_edge"][ch_idx] = [0, 0]
        if mesh.cbflg[p1_idx] and mesh.cbflg[p2_idx]:  # cavity boundary
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
    u = np.array([[ns.u[mesh.pmap[i]] if mesh.pmap[i] > -1 else np.zeros(1)] for i in range(mesh.tri.npoints)]).reshape(mesh.tri.npoints).flatten()
    v = np.array([[ns.v[mesh.pmap[i]] if mesh.pmap[i] > -1 else np.zeros(1)] for i in range(mesh.tri.npoints)]).reshape(mesh.tri.npoints).flatten()
    w = np.array([[ns.w[mesh.pmap[i]] if mesh.pmap[i] > -1 else np.zeros(1)] for i in range(mesh.tri.npoints)]).reshape(mesh.tri.npoints).flatten()
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
    u_tot = np.sqrt(u**2 + v**2 + w**2)
    mask = u_tot > 0
    u_norm = np.zeros_like(u, dtype=np.float32)
    v_norm = np.zeros_like(v, dtype=np.float32)
    w_norm = np.zeros_like(w, dtype=np.float32)
    u_norm[mask] = u[mask] / u_tot[mask]
    v_norm[mask] = v[mask] / u_tot[mask]
    w_norm[mask] = w[mask] / u_tot[mask]

    var = np.array(u, copy=True)

    grid_x, grid_y, grid_z = np.mgrid[x.min():x.max():100j, y.min():y.max():100j, z.min():z.max():100j]
    u_intp = sp.interpolate.griddata(mesh.tri.points, var, (grid_x, grid_y, grid_z), method='linear')
    lrange = np.linspace(var.min(), var.max(), 6)
    cmap = cm.coolwarm

    plt.rcParams['figure.figsize'] = 16, 6
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    zidx = 25
    label = r'$u(z=L_z/4)$'
    surf = ax.contourf(grid_x[:, :, zidx], grid_y[:, :, zidx], u_intp[:, :, zidx], cmap=cmap)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=10, alpha=0.5)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('auto')
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')

    ax = fig.add_subplot(1, 3, 2)
    zidx = 50
    label = r'$u(z=L_z/2)$'
    surf = ax.contourf(grid_x[:, :, zidx], grid_y[:, :, zidx], u_intp[:, :, zidx], cmap=cmap)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=10, alpha=0.5)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('auto')
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')

    ax = fig.add_subplot(1, 3, 3)
    zidx = 75
    label = r'$u(z=3L_z/4)$'
    surf = ax.contourf(grid_x[:, :, zidx], grid_y[:, :, zidx], u_intp[:, :, zidx], cmap=cmap)
    ax.quiver(x[::5], y[::5], u_norm[::5], v_norm[::5], units='xy', scale=10, alpha=0.5)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('auto')
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.1, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')

    plt.tight_layout()
    plt.savefig('NS_3D_ex002.png')
    plt.close()
# FEM 3D structural mechanics equation solver
# Trapezoidal beam with traction

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

from trapezoid_3d import create_nodes, create_elements
from FEM_mesh_3d import Mesh_from_FreeCAD
from FEM_SM_3d import FESM3D

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
        return np.array([0., - 1.2e3 * 9.8, 0]) # gravity in N/m3

    # Domain space
    iteration_max = 1
    T = 2e4 # surface force in N/m

    """ 
    Create a rectangular mesh
    """
    print('Create a mesh')
    # Create a mesh object with Delaunay mesh and a boundary condition dictionary
    femmesh = FemMesh()
    exitCode = create_nodes(femmesh)
    exitCode = create_elements(femmesh)
    simplices_coord = np.round(np.array(femmesh.faces, dtype=np.float32)[:, :], 2) / 100
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
            mesh.bc_points_u["dirichlet"][point_index] = 0
            mesh.bc_points_v["dirichlet"][point_index] = 0
            mesh.bc_points_w["dirichlet"][point_index] = 0
            
    # Set the Neumann boundary conditions
    for ch_idx, edge_points in enumerate(mesh.convex_hull):
        p1_idx, p2_idx = edge_points
        p1, p2 = mesh.tri.points[edge_points]
        if p1[0] == x_min and p2[0] == x_min:  # West
            pass
        elif p1[2] == z_min and p2[2] == z_min:  # Bottom
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_w["neumann_edge"][ch_idx] = [0, 0]
        elif p1[2] == z_max and p2[2] == z_max:  # Top
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_w["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_min and p2[1] == y_min:  # South
            mesh.bc_points_u["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_w["neumann_edge"][ch_idx] = [0, 0]
        elif p1[1] == y_max and p2[1] == y_max:  # North
            mesh.bc_points_u["neumann_edge"][ch_idx] = [T, T]
            mesh.bc_points_v["neumann_edge"][ch_idx] = [0, 0]
            mesh.bc_points_w["neumann_edge"][ch_idx] = [0, 0]
        else:
            mesh.bc_points_u["neumann_edge"][ch_idx] = [T/np.sqrt(2.0), T/np.sqrt(2.0)] # T/np.sqrt(2.0)
            mesh.bc_points_v["neumann_edge"][ch_idx] = [-T/np.sqrt(2.0), -T/np.sqrt(2.0)]
            mesh.bc_points_w["neumann_edge"][ch_idx] = [0, 0]

    """ 
    Solve the 3D SM problem
    """
    print('Solve the 3D SM problem')
    # Set a simulation object
    sm = FESM3D(mesh, _f=RHS, _nu=0.499, _E=0.1e9, _rho=1.2e3, _gpu=True, _sparse=False, _dt=0.001) # rubber

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
    
    ux = u[0::3, 0]
    uy = u[1::3, 0]
    uz = u[2::3, 0]
    u_tot = np.sqrt(ux**2+uy**2+uz**2)
    ux_norm = ux / u_tot
    uy_norm = uy / u_tot
    uz_norm = uz / u_tot
    x_new = x + ux
    y_new = y + uy
    z_new = z + uz
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(x, z, y, s=2, color='b')
    ax.scatter(x_new, z_new, y_new, s=4, color='g')
    ax.view_init(elev=0, azim=-90, roll=0)
    ax.set_xlim([x_min, x_max])
    ax.set_zlim([y_min, y_max])
    ax.set_ylim([z_min, z_max])
    ax.set_title('Solution')
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(x, z, y, s=2, color='b')
    ax.scatter(x[mesh.tri.boundary_points], z[mesh.tri.boundary_points], y[mesh.tri.boundary_points], s=10, color='k')
    ax.view_init(elev=30, azim=+50, roll=0)
    ax.set_xlim([x_min, x_max])
    ax.set_zlim([y_min, y_max])
    ax.set_ylim([z_min, z_max])
    ax.set_title('Grid')
    
    plt.tight_layout()
    plt.savefig('SM_3D_ex001.png')
    plt.close()
# class
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib as mpl
import matplotlib.cm as cm

def plot3d(ax, data, varmin, varmax, x, y):
    var = np.array(data, copy=True)
    varmin = 0
    varmax = 1
    lrange = np.linspace(varmin, varmax, 11)
    var[var < varmin] = varmin
    var[var > varmax] = varmax
    
    ax.view_init(elev=20, azim=130, roll=0)
    
    surf = ax.plot_trisurf(x, y, var,
                           linewidth=0.2, antialiased=True, cmap=cm.coolwarm, label="FE")
    
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.02, ticks=lrange)
    cbar.ax.set_xlabel('u')
    cbar.ax.xaxis.set_label_position('bottom')

# Mesh class
class Mesh:
    def __init__(self, x_min, x_max, n_x, y_min, y_max, n_y):
        # Create a list with points coordinate (x,y)
        points = []
        nodes_x = np.linspace(x_min, x_max, n_x)
        nodes_y = np.linspace(y_min, y_max, n_y)
        for x in nodes_x:
            for y in nodes_y:
                # points.append([x, y])
                if x**2 + y**2 <= (x_max)**2:
                    points.append([x, y])
        points = np.array(points)
        self.points = points

        # Create Delaunay object
        self.tri = sp.spatial.Delaunay(points)

        # Identify the boundary points
        self.boundary_points = np.unique(self.tri.convex_hull.flatten())

        # Initialize the boundary conditions dictionary
        self.bc_points = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }

# Triangle class
class GenericTriElement:
    def __init__(self):
        # Shape functions derivatives in a local triangular element
        N1_dxi = 1
        N2_dxi = 0
        N3_dxi = -1
        N1_deta = 0
        N2_deta = 1
        N3_deta = -1
        self.dN = np.array([
            [N1_dxi, N1_deta],
            [N2_dxi, N2_deta],
            [N3_dxi, N3_deta]])

    @staticmethod
    def N1(xi, eta):
        return xi

    @staticmethod
    def N2(xi, eta):
        return eta

    @staticmethod
    def N3(xi, eta):
        return 1 - xi - eta

    # Coordinate transformation local to global
    def get_xy(self, xi, eta, p1, p2, p3):
        return (p1[0] * self.N1(xi, eta) + p2[0] * self.N2(xi, eta) + p3[0] * self.N3(xi, eta),
                p1[1] * self.N1(xi, eta) + p2[1] * self.N2(xi, eta) + p3[1] * self.N3(xi, eta))

# Gaussian integration class
class GaussianQuadratureTri:
    def __init__(self):
        # nip = 3 # number of integration points
        self.wps = [(0.5, 0.5), (0.5, 0), (0, 0.5)]  # weighted points
        self.ws = (1 / 6, 1 / 6, 1 / 6)              # weights
        self.tri_element = GenericTriElement()

    # Calculate the numerical integration for each node
    def calculate(self, _f, p1, p2, p3):
        # Get the global (x,y) coordinates at the weighted points
        xys = [self.tri_element.get_xy(wp[0], wp[1], p1, p2, p3) for wp in self.wps]

        return np.array([
            sum([w * _f(xy[0], xy[1]) * self.tri_element.N1(
                wp[0], wp[1]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1]) * self.tri_element.N2(
                wp[0], wp[1]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1]) * self.tri_element.N3(
                wp[0], wp[1]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
        ])

# FEM Poisson 2D solver class
class FEheat2D:
    def __init__(self, _mesh, _f, _u=None):
        self.gte = GenericTriElement()
        self.gauss_quad = GaussianQuadratureTri()

        self.mesh = _mesh
        self.n_elements = self.mesh.tri.nsimplex
        self.n_points = self.mesh.tri.npoints

        self.f = _f

        self.M = np.zeros((self.n_points, self.n_points))
        self.Minv = np.zeros_like(self.M)
        self.K = np.zeros_like(self.M)
        self.I = np.eye(self.n_points)

        self.A = np.zeros((self.n_points, self.n_points))
        self.b = np.zeros((self.n_points, 1))
        self.s = np.zeros_like(self.b)
        if _u is None:
            self.u = np.zeros_like(self.b)
        else:
            self.u = np.array(_u, copy=True)

        self.dt = 1e-3

    # @staticmethod
    def calc_local_update(self, p1, p2, p3):
        # Calculate the Jacobian, its determinant, and inverse
        j11 = p1[0] - p3[0]  # x_1 - x_3
        j12 = p1[1] - p3[1]  # y_1 - y_3
        j21 = p2[0] - p3[0]  # x_2 - x_3
        j22 = p2[1] - p3[1]  # y_2 - y_3
        j_det = j11 * j22 - j21 * j12
        j_inv = np.array([[j22, -j12],
                          [-j21, j11]]) / j_det

        # Calculate matrix solution of one element
        K_local = np.zeros((3, 3))
        b_local = np.zeros((3, 1))

        # local stiffness matrix
        K_local[0, 0] = (0.5 / j_det) * ( (p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 )
        K_local[1, 1] = (0.5 / j_det) * ( (p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 )
        K_local[2, 2] = (0.5 / j_det) * ( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

        K_local[1, 0] = K_local[0, 1] = (0.5 / j_det) * ( (p3[0] - p2[0]) * (p1[0] - p3[0]) + (p3[1] - p2[1]) * (p1[1] - p3[1]) )
        K_local[2, 0] = K_local[0, 2] = (0.5 / j_det) * ( (p3[0] - p2[0]) * (p2[0] - p1[0]) + (p3[1] - p2[1]) * (p2[1] - p1[1]) )
        K_local[1, 2] = K_local[2, 1] = (0.5 / j_det) * ( (p1[0] - p3[0]) * (p2[0] - p1[0]) + (p1[1] - p3[1]) * (p2[1] - p1[1]) )

        # local mass matrix
        M_local = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) * (j_det / 24.0)

        # b matrix
        b_local = j_det * self.gauss_quad.calculate(self.f, p1, p2, p3)

        return K_local, M_local, b_local

    def set_K_M(self):
        # Calculate the global matrix solution
        for i, el_ps in enumerate(self.mesh.tri.simplices):
            # Extract element's nodes
            p1, p2, p3 = (self.mesh.tri.points[el_ps[0]],
                          self.mesh.tri.points[el_ps[1]],
                          self.mesh.tri.points[el_ps[2]])

            # Store local element's solution
            K_local, M_local, _ = self.calc_local_update(p1, p2, p3)

            # Assemble element's matrix solution into global matrix
            columns = np.array([el_ps for _ in range(3)])
            rows = columns.T
            self.K[rows, columns] += K_local
            self.M[rows, columns] += M_local

    def set_s(self):
        # Calculate the global matrix solution
        for i, el_ps in enumerate(self.mesh.tri.simplices):
            # Extract element's nodes
            p1, p2, p3 = (self.mesh.tri.points[el_ps[0]],
                          self.mesh.tri.points[el_ps[1]],
                          self.mesh.tri.points[el_ps[2]])

            # Store local element's solution
            _, _, b_local = self.calc_local_update(p1, p2, p3)
            # print(M_local)

            # Assemble element's matrix solution into global matrix
            self.s[el_ps, 0] += b_local


    def set_boundary_conditions_dirichlet(self):
        # Set Dirichlet boundary conditions
        u_temp = np.zeros_like(self.b)
        for key, value in self.mesh.bc_points["dirichlet"].items():
            u_temp[key] = value
        self.b -= self.A @ u_temp

    def set_boundary_conditions_neumann(self):
        # Set Neumann boundary conditions
        for ch_idx, du_values in self.mesh.bc_points["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.tri.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.s[p_idx] += 0.5 * distance * du_value  # du_boundary

    def initialze(self):
        
        # Initialize the K, M and b
        self.K = np.zeros((self.n_points, self.n_points))
        self.M = np.zeros((self.n_points, self.n_points))
        self.b = np.zeros((self.n_points, 1))
        
        # Calculate K and M entries
        self.set_K_M()

        # Inverse mass matrix
        self.Minv = np.linalg.inv(self.M)

        # Calculate A entries
        self.A = self.I + (self.dt/2.0) * self.Minv @ self.K

         # print('Evaluate source matrix')
        self.set_s()

        # apply boundary conditions Neumann
        self.set_boundary_conditions_neumann()

    def solve(self):

        # RHS
        self.b = self.dt * self.Minv @ self.s + (self.I - (self.dt/2.0) * self.Minv @ self.K) @ self.u

        # apply boundary conditions Dirichlet
        self.set_boundary_conditions_dirichlet()  

        # Exclude known u from the Dirichlet boundary condition
        points_to_solve = []
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points["dirichlet"]:
                points_to_solve.append(p_idx)
                
        # Solve u = A^-1 * b
        self.u[points_to_solve] = sp.linalg.solve(self.A[points_to_solve, :][:, points_to_solve], self.b[points_to_solve])
        # self.u[points_to_solve, 0], exitCode = sp.sparse.linalg.gmres(self.A[points_to_solve, :][:, points_to_solve], self.b[points_to_solve])

        # Set the known
        for key, value in self.mesh.bc_points["dirichlet"].items():
            self.u[key] = value
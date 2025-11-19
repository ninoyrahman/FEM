# class
import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy as cps

# Mesh class
class Mesh:
    def __init__(self, x_min, x_max, n_x, y_min, y_max, y_n):
        # Create a list with points coordinate (x,y)
        points = []
        nodes_x = np.linspace(x_min, x_max, n_x)
        nodes_y = np.linspace(y_min, y_max, y_n)
        for nx in nodes_x:
            for ny in nodes_y:
                points.append([nx, ny])
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
class FEPoisson2D:
    def __init__(self, _mesh, _f, _gpu=False, _sparse=False):
        self.gte = GenericTriElement()
        self.gauss_quad = GaussianQuadratureTri()

        self.mesh = _mesh
        self.n_elements = self.mesh.tri.nsimplex
        self.n_points = self.mesh.tri.npoints

        self.f = _f

        self.A = np.zeros((self.n_points, self.n_points))
        self.b = np.zeros((self.n_points, 1))
        self.u = np.zeros_like(self.b)

        self.points_to_solve = np.array([], dtype=np.int32)

        self.sparse = _sparse
        self.gpu = _gpu
        if self.gpu:
            # Memory Pools for Efficient Allocation
            self.mp = cp.get_default_memory_pool()
            self.pp = cp.get_default_pinned_memory_pool()
            
            self.A_d = cp.zeros((self.n_points, self.n_points))
            self.b_d = cp.zeros((self.n_points, 1))
            self.u_d = cp.zeros_like(self.b_d)
            self.points_to_solve_d = np.array([], dtype=np.int32)
            
    # @staticmethod
    def calc_local_update_Ab(self, p1, p2, p3):
        # Calculate the Jacobian, its determinant, and inverse
        j11 = p1[0] - p3[0]  # x_1 - x_3
        j12 = p1[1] - p3[1]  # y_1 - y_3
        j21 = p2[0] - p3[0]  # x_2 - x_3
        j22 = p2[1] - p3[1]  # y_2 - y_3
        j_det = j11 * j22 - j21 * j12
        j_inv = np.array([[j22, -j12],
                          [-j21, j11]]) / j_det

        # Calculate matrix solution of one element
        A_local = np.zeros((3, 3))
        b_local = np.zeros((3, 1))
        for j in range(3):
            A_local[j, j] = 0.5 * np.dot(j_inv, self.gte.dN[j, :]
                                         ) @ np.dot(j_inv, self.gte.dN[j, :]) * j_det
        A_local[1, 0] = A_local[0, 1] = 0.5 * np.dot(
            j_inv, self.gte.dN[1, :]) @ np.dot(j_inv, self.gte.dN[0, :]) * j_det
        A_local[2, 0] = A_local[0, 2] = 0.5 * np.dot(
            j_inv, self.gte.dN[2, :]) @ np.dot(j_inv, self.gte.dN[0, :]) * j_det
        A_local[1, 2] = A_local[2, 1] = 0.5 * np.dot(
            j_inv, self.gte.dN[2, :]) @ np.dot(j_inv,self.gte.dN[1, :]) * j_det
        b_local = j_det * self.gauss_quad.calculate(self.f, p1, p2, p3)

        return A_local, b_local

    def set_A_b(self):
        # Calculate the global matrix solution
        for i, el_ps in enumerate(self.mesh.tri.simplices):
            # Extract element's nodes
            p1, p2, p3 = (self.mesh.tri.points[el_ps[0]],
                          self.mesh.tri.points[el_ps[1]],
                          self.mesh.tri.points[el_ps[2]])

            # Store local element's solution
            A_local, b_local = self.calc_local_update_Ab(p1, p2, p3)

            # Assemble element's matrix solution into global matrix
            columns = np.array([el_ps for _ in range(3)])
            rows = columns.T
            self.A[rows, columns] += A_local
            self.b[el_ps, 0] += b_local

    def set_boundary_conditions(self):
        # Set Dirichlet boundary conditions
        u_temp = np.zeros_like(self.b)
        for key, value in self.mesh.bc_points["dirichlet"].items():
            u_temp[key] = value
        self.b -= self.A @ u_temp

        # Set Neumann boundary conditions
        for ch_idx, du_values in self.mesh.bc_points["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.tri.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.b[p_idx] += 0.5 * distance * du_value  # du_boundary

    def process(self):
        # Initialize the A and b
        self.A = np.zeros((self.n_points, self.n_points))
        self.b = np.zeros((self.n_points, 1))

        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points["dirichlet"]:
                self.points_to_solve = np.append(self.points_to_solve, p_idx)

        # print(self.points_to_solve)
        if self.gpu:
            self.points_to_solve_d = cp.asarray(self.points_to_solve)

        # Calculate A and b entries
        self.set_A_b()

        # Set boundary conditions
        self.set_boundary_conditions()

    def solve(self):
        # Initialize u
        self.u = np.zeros_like(self.b)

        if self.gpu:
            print('Solving using GPU')

            # host to device data transfer
            cp.cuda.runtime.memcpy(self.A_d.data.ptr, self.A.ctypes.data, self.A.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.b_d.data.ptr, self.b.ctypes.data, self.b.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            # solve
            if self.sparse:
                print('Solving sparse matrix')
                A_d_sparse = cps.sparse.csr_matrix(self.A_d[self.points_to_solve_d, :][:, self.points_to_solve_d])
                self.u_d[self.points_to_solve_d] = cps.sparse.linalg.spsolve(A_d_sparse, self.b_d[self.points_to_solve_d])
            else: 
                self.u_d[self.points_to_solve_d] = cp.linalg.solve(self.A_d[self.points_to_solve_d, :][:, self.points_to_solve_d], self.b_d[self.points_to_solve_d])
            
            # device to host data transfer
            cp.cuda.runtime.memcpy(self.u.ctypes.data, self.u_d.data.ptr, self.u_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)

            # Inspect pool usage
            # print(f"Used: {self.mp.used_bytes() / 1e6:.2f} MB")
            # print(f"Total allocated: {self.mp.total_bytes() / 1e6:.2f} MB")
            
            # Free unused blocks back to OS
            self.mp.free_all_blocks()
        else:
            print('Solving using CPU')
            if self.sparse:
                print('Solving sparse matrix')
                A_sparse = sp.sparse.csr_matrix(self.A[self.points_to_solve, :][:, self.points_to_solve])
                self.u[self.points_to_solve, 0] = sp.sparse.linalg.spsolve(A_sparse, self.b[self.points_to_solve])
            else:
                self.u[self.points_to_solve] = sp.linalg.solve(self.A[self.points_to_solve, :][:, self.points_to_solve], self.b[self.points_to_solve])

        # Set the known
        for key, value in self.mesh.bc_points["dirichlet"].items():
            self.u[key] = value
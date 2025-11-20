# class
import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy as cps
from FEM_tri import GenericTriElement, GaussianQuadratureTri

# FEM Poisson 2D solver class
class FEPoisson2D:
    """
    class for 2D Poisson equation solver with finite element method

    ...

    Attributes
    ----------
        gte = GenericTriElement
            class for 2D basis for a triangular element
        gauss_quad = GaussianQuadratureTri
            Gaussian integration class
        mesh = Mesh
            mesh for computational domain
        n_elements = int 
            number of simplex in Delaunay triangulation
        n_points = int
            number of simplex in Delaunay triangulation
        f = function
            R.H.S function
        A = numpy.ndarray
            A matrix of A u = b
        b = numpy.ndarray
            b vector of A u = b
        u = numpy.ndarray
            solution of A u = b
        points_to_solve = numpy.ndarray
            index of points to solve for u
        sparse = bool
            True: use sparse matrix solver, False: use dense matrix solver
        gpu = bool
            True: use GPU matrix solver, False: use CPU matrix solver
        mp = CuPy function
            get default memory pool
        pp = CuPy function
            get default pinned memory pool    
        A_d = cupy.ndarray
            GPU A matrix of A u = b
        b_d = cupy.ndarray
            GPU b vector of A u = b
        u_d = cupy.ndarray
            GPU solution of A u = b
        
    Methods
    -------
    calc_local_update_Ab(self, p1, p2, p3)
        Calculate the Jacobian, its determinant, and inverse
    set_A_b(self)
        Calculate the global matrix solution
    set_boundary_conditions(self):
        Set Dirichlet boundary conditions
    set_boundary_conditions(self):
        Set Dirichlet boundary conditions
    process(self):
        Initialize the A and b
    solve(self):
        Solve A u = b    
    """      
    def __init__(self, _mesh, _f, _gpu=False, _sparse=False):
        """
        Parameters
        ----------
        _mesh : Mesh
            mesh for computational domain
        _f = function
            R.H.S function
        _gpu = bool
            True: use GPU matrix solver, False: use CPU matrix solver, default CPU
        _sparse = bool
            True: use sparse matrix solver, False: use dense matrix solver, default Dense
        """        
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
        """
        Parameters
        ----------
        p1, p2, p3: numpy.ndarray
            coordinates of a triangle
        """        
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
        """
        Parameters
        ----------
        """        
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
        """
        Parameters
        ----------
        """        
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
        """
        Parameters
        ----------
        """           
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
        """
        Parameters
        ----------
        """        
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
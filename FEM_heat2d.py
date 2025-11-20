# FEM_heat2d
import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy as cps
from FEM_tri import GenericTriElement, GaussianQuadratureTri
from FEM_mesh import Mesh

# FEM Poisson 2D solver class
class FEheat2D:
"""
    class for 2D Poisson equation solver with finite element method

    ...

    Attributes
    ----------
        gte : GenericTriElement
            Class for 2D basis for a triangular element
        gauss_quad : GaussianQuadratureTri
            Gaussian integration class
        mesh : Mesh
            Mesh for computational domain
        n_elements : int 
            Number of simplex in Delaunay triangulation
        n_points : int
            Number of simplex in Delaunay triangulation
        f : function
            R.H.S function
        dt : float
            Time step size for integration
        M : numpy.ndarray
            Mass matrix
        Minv : numpy.ndarray
            Inverse of mass matrix
        K : numpy.ndarray
            Stiffness matrix
        I : numpy.ndarray
            Identity matrix
        s : numpy.ndarray
            Source vector
        A : numpy.ndarray
            A matrix of A u = b
        b : numpy.ndarray
            b vector of A u = b
        u : numpy.ndarray
            Solution of A u = b
        points_to_solve : numpy.ndarray
            Index of points to solve for u
        sparse : bool
            True: use sparse matrix solver, False: use dense matrix solver
        gpu : bool
            True: use GPU matrix solver, False: use CPU matrix solver
        mp : CuPy function
            Get default memory pool
        pp : CuPy function
            Get default pinned memory pool    
        A_d : cupy.ndarray
            GPU A matrix of A u = b
        b_d : cupy.ndarray
            GPU b vector of A u = b
        u_d : cupy.ndarray
            GPU solution of A u = b
        
    Methods
    -------
    calc_local_update(self, p1, p2, p3):
        Calculate the Jacobian, its determinant, and inverse
    set_K_M(self):
        Calculate the global mass and stiffness matrix
    set_s(self):
        Calculate the global source vector
    set_boundary_conditions_dirichlet(self):
        Set Dirichlet boundary conditions
    set_boundary_conditions_neumann(self):
        Set Neumann boundary conditions
    initialze(self):
        Initialize the mass, stiffness matrix and source vector
    solve(self):
        Solve A u = b    
    """    
    def __init__(self, _mesh, _f, _u=None, _gpu=False, _sparse=False):
        """
        Parameters
        ----------
        _mesh : Mesh
            Mesh for computational domain
        _f : function
            R.H.S function
        _u : numpy.ndarray
            Initial guess
        _gpu : bool
            True: use GPU matrix solver, False: use CPU matrix solver, default CPU
        _sparse : bool
            True: use sparse matrix solver, False: use dense matrix solver, default Dense
        """         
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

        self.points_to_solve = np.array([], dtype=np.int32)

        self.gpu = _gpu
        self.sparse = _sparse
        if self.gpu:
            # Memory Pools for Efficient Allocation
            self.mp = cp.get_default_memory_pool()
            self.pp = cp.get_default_pinned_memory_pool()
            
            self.A_d = cp.zeros((self.n_points, self.n_points))
            self.b_d = cp.zeros((self.n_points, 1))
            self.u_d = cp.zeros_like(self.b_d)
            self.points_to_solve_d = np.array([], dtype=np.int32)

        print('Solving using GPU:', self.gpu)
        print('Solving using sparse matrix:', self.sparse)

    # @staticmethod
    def calc_local_update(self, p1, p2, p3):
        """
        Parameters
        ----------
        p1, p2, p3: numpy.ndarray
            Coordinates of a triangle
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
        """
        Parameters
        ----------
        """     
        # Calculate the global mass and stiffness matrix
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
        """
        Parameters
        ----------
        """        
        # Calculate the global source vector
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
        """
        Parameters
        ----------
        """        
        # Set Dirichlet boundary conditions
        u_temp = np.zeros_like(self.b)
        for key, value in self.mesh.bc_points["dirichlet"].items():
            u_temp[key] = value
        self.b -= self.A @ u_temp

    def set_boundary_conditions_neumann(self):
        """
        Parameters
        ----------
        """        
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
        """
        Parameters
        ----------
        p1, p2, p3: numpy.ndarray
            coordinates of a triangle
        """        
        # Initialize the mass, stiffness matrix and source vector
        self.K = np.zeros((self.n_points, self.n_points))
        self.M = np.zeros((self.n_points, self.n_points))
        self.b = np.zeros((self.n_points, 1))

        # assign points to solve
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points["dirichlet"]:
                self.points_to_solve = np.append(self.points_to_solve, p_idx)

        if self.gpu:
            self.points_to_solve_d = cp.asarray(self.points_to_solve)        
        
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
        """
        Parameters
        ----------
        """
        # RHS
        self.b = self.dt * self.Minv @ self.s + (self.I - (self.dt/2.0) * self.Minv @ self.K) @ self.u

        # apply boundary conditions Dirichlet
        self.set_boundary_conditions_dirichlet()
                
        # Solve u = A^-1 * b
        if not self.gpu:
            if self.sparse:
                A_sparse = sp.sparse.csr_matrix(self.A[self.points_to_solve, :][:, self.points_to_solve])
                # self.u[self.points_to_solve, 0] = sp.sparse.linalg.spsolve(A_sparse, self.b[self.points_to_solve])
                self.u[self.points_to_solve, 0], exitCode = sp.sparse.linalg.gmres(A_sparse, self.b[self.points_to_solve], x0=self.u[self.points_to_solve, 0])
            else:
                self.u[self.points_to_solve] = sp.linalg.solve(self.A[self.points_to_solve, :][:, self.points_to_solve], self.b[self.points_to_solve])
        else:
            # host to device data transfer
            cp.cuda.runtime.memcpy(self.A_d.data.ptr, self.A.ctypes.data, self.A.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.b_d.data.ptr, self.b.ctypes.data, self.b.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            # solve
            if self.sparse:
                A_d_sparse = cps.sparse.csr_matrix(self.A_d[self.points_to_solve_d, :][:, self.points_to_solve_d])
                # self.u_d[self.points_to_solve_d] = cps.sparse.linalg.spsolve(A_d_sparse, self.b_d[self.points_to_solve_d])
                self.u_d[self.points_to_solve_d, 0], exitCode = cps.sparse.linalg.gmres(A_d_sparse, self.b_d[self.points_to_solve_d], x0=self.u_d[self.points_to_solve_d, 0])
            else: 
                self.u_d[self.points_to_solve_d] = cp.linalg.solve(self.A_d[self.points_to_solve_d, :][:, self.points_to_solve_d], self.b_d[self.points_to_solve_d])
            
            # device to host data transfer
            cp.cuda.runtime.memcpy(self.u.ctypes.data, self.u_d.data.ptr, self.u_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)

            # Inspect pool usage
            # print(f"Used: {self.mp.used_bytes() / 1e6:.2f} MB")
            # print(f"Total allocated: {self.mp.total_bytes() / 1e6:.2f} MB")
            
            # Free unused blocks back to OS
            self.mp.free_all_blocks()

        # Set the known
        for key, value in self.mesh.bc_points["dirichlet"].items():
            self.u[key] = value
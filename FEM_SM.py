# FEM_NS
import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy as cps
from FEM_tri import GenericTriElement, GaussianQuadratureTri

# FEM SM 2D solver class


class FESM2D:
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
            Number of points in Delaunay triangulation
        f : function
            R.H.S function
        dt : float
            Time step size for integration
        I : numpy.ndarray
            Identity matrix
        M : numpy.ndarray
            Mass matrix
        Minv : numpy.ndarray
            Inverse of mass matrix
        K : numpy.ndarray
            Stiffness matrix
        A : numpy.ndarray
            A = Minv * K
        s : numpy.ndarray
            Source vector
        b : numpy.ndarray
            RHS
        u : numpy.ndarray
            displacement
        u_dirichlet : numpy.ndarray
            Dirichlet boundary values

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
        I_d : cupy.ndarray
            GPU I matrix
        K_d : cupy.ndarray
            GPU K matrix
        A_d : cupy.ndarray
            GPU A matrix
        b_d : cupy.ndarray
            GPU b vector
        u_d : cupy.ndarray
            GPU displacement
        u_dirichlet_d : cupy.ndarray
            GPU u_dirichlet
        points_to_solve_d : cupy.ndarray
            GPU points_to_solve

    Methods
    -------
    time_step_size(self):
        Integration time step determination
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
        Solve Navier Stokes equations
    """

    def __init__(self, _mesh, _f, _nu=0.3, _plain_stress=True, _u=None, _gpu=False, _sparse=False, _dt=0.001):
        """
        Parameters
        ----------
        _mesh : Mesh
            Mesh for computational domain
        _f : function
            R.H.S function
        _u : numpy.ndarray
            Initial guess for displacement
        _gpu : bool
            True: use GPU matrix solver, False: use CPU matrix solver, default CPU
        _sparse : bool
            True: use sparse matrix solver, False: use dense matrix solver, default Dense
        _dt : float
            Integration time step
        """
        self.gte = GenericTriElement()
        self.gauss_quad = GaussianQuadratureTri()

        self.mesh = _mesh
        self.n_elements = self.mesh.tri.nsimplex
        self.n_points = self.mesh.tri.npoints

        self.f = _f

        self.M = np.zeros((2*self.n_points, 2*self.n_points))
        self.Minv = np.zeros_like(self.M)
        self.K = np.zeros_like(self.M)
        self.A = np.zeros_like(self.M)
        self.I = np.eye(2*self.n_points)

        self.b = np.zeros((2*self.n_points, 1))
        self.s = np.zeros_like(self.b)
        self.u_dirichlet = np.zeros_like(self.b)

        if _u is None:
            self.u = np.zeros_like(self.b)
        else:
            self.u = np.array(_u, copy=True)

        self.dt = _dt
        self.nu = _nu
        self.plain_stress = _plain_stress

        self.points_to_solve = np.array([], dtype=np.int32)

        self.gpu = _gpu
        self.sparse = _sparse
        if self.gpu:
            # Memory Pools for Efficient Allocation
            self.mp = cp.get_default_memory_pool()
            self.pp = cp.get_default_pinned_memory_pool()

            self.K_d = cp.zeros((2*self.n_points, 2*self.n_points))
            self.I_d = cp.zeros_like(self.K_d)
            self.A_d = cp.zeros_like(self.K_d)

            self.b_d = cp.zeros((2*self.n_points, 1))

            self.u_d = cp.zeros_like(self.b_d)
            self.u_dirichlet_d = cp.zeros_like(self.b_d)

            self.points_to_solve_d = cp.array([], dtype=cp.int32)

        print('Solving using GPU:', self.gpu)
        print('Solving using sparse matrix:', self.sparse)
        print('Solving for nu:', self.nu)
        if self.plain_stress:
            print('Solving for plain stress case')
        else:
            print('Solving for plain strain case')

    def time_step_size(self):
        """
        Parameters
        ----------
        """
        # time step determination
        dmin2 = 1.0
        for i, el_ps in enumerate(self.mesh.tri.simplices):
            p1, p2, p3 = (self.mesh.tri.points[el_ps[0]],
                          self.mesh.tri.points[el_ps[1]],
                          self.mesh.tri.points[el_ps[2]])

            d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
            d32 = (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2
            dmin2 = min(dmin2, d12, d13, d32)

        self.dt = dmin2

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
        K_local = np.zeros((6, 6))
        b_local = np.zeros((6, 1))

        # local stiffness matrix
        if self.plain_stress:
            K_local[0, 0] = (self.nu*p2[0]**2 - 2*self.nu*p2[0]*p3[0] + self.nu*p3[0]**2 - p2[0]**2 + 2 *
                             p2[0]*p3[0] - p3[0]**2 - 2*p2[1]**2 + 4*p2[1]*p3[1] - 2*p3[1]**2)/(4*(self.nu**2 - 1)) / j_det
            K_local[0, 1] = K_local[1, 0] = (
                p2[0]*p2[1] - p2[0]*p3[1] - p3[0]*p2[1] + p3[0]*p3[1])/(4*(self.nu - 1)) / j_det
            K_local[0, 2] = K_local[2, 0] = (-self.nu*p1[0]*p2[0] + self.nu*p1[0]*p3[0] + self.nu*p2[0]*p3[0] - self.nu*p3[0]**2 + p1[0]*p2[0] -
                                             p1[0]*p3[0] - p2[0]*p3[0] + p3[0]**2 + 2*p1[1]*p2[1] - 2*p1[1]*p3[1] - 2*p2[1]*p3[1] + 2*p3[1]**2)/(4*(self.nu**2 - 1)) / j_det
            K_local[0, 3] = K_local[3, 0] = (-2*self.nu*p1[0]*p2[1] + 2*self.nu*p1[0]*p3[1] + self.nu*p2[0]*p1[1] - self.nu*p2[0]*p3[1] - self.nu*p3[0]
                                             * p1[1] + 2*self.nu*p3[0]*p2[1] - self.nu*p3[0]*p3[1] - p2[0]*p1[1] + p2[0]*p3[1] + p3[0]*p1[1] - p3[0]*p3[1])/(4*(self.nu**2 - 1)) / j_det
            K_local[0, 4] = K_local[4, 0] = (self.nu*p1[0]*p2[0] - self.nu*p1[0]*p3[0] - self.nu*p2[0]**2 + self.nu*p2[0]*p3[0] - p1[0]*p2[0] +
                                             p1[0]*p3[0] + p2[0]**2 - p2[0]*p3[0] - 2*p1[1]*p2[1] + 2*p1[1]*p3[1] + 2*p2[1]**2 - 2*p2[1]*p3[1])/(4*(self.nu**2 - 1)) / j_det
            K_local[0, 5] = K_local[5, 0] = (2*self.nu*p1[0]*p2[1] - 2*self.nu*p1[0]*p3[1] - self.nu*p2[0]*p1[1] - self.nu*p2[0]*p2[1] + 2*self.nu*p2[0]
                                             * p3[1] + self.nu*p3[0]*p1[1] - self.nu*p3[0]*p2[1] + p2[0]*p1[1] - p2[0]*p2[1] - p3[0]*p1[1] + p3[0]*p2[1])/(4*(self.nu**2 - 1)) / j_det

            K_local[1, 1] = (self.nu*p2[1]**2 - 2*self.nu*p2[1]*p3[1] + self.nu*p3[1]**2 - 2*p2[0]**2 +
                             4*p2[0]*p3[0] - 2*p3[0]**2 - p2[1]**2 + 2*p2[1]*p3[1] - p3[1]**2)/(4*(self.nu**2 - 1)) / j_det
            K_local[1, 2] = K_local[2, 1] = (self.nu*p1[0]*p2[1] - self.nu*p1[0]*p3[1] - 2*self.nu*p2[0]*p1[1] + 2*self.nu*p2[0]*p3[1] + 2*self.nu*p3[0]
                                             * p1[1] - self.nu*p3[0]*p2[1] - self.nu*p3[0]*p3[1] - p1[0]*p2[1] + p1[0]*p3[1] + p3[0]*p2[1] - p3[0]*p3[1])/(4*(self.nu**2 - 1)) / j_det
            K_local[1, 3] = K_local[3, 1] = (-self.nu*p1[1]*p2[1] + self.nu*p1[1]*p3[1] + self.nu*p2[1]*p3[1] - self.nu*p3[1]**2 + 2*p1[0]*p2[0] - 2*p1[0]*p3[0] - 2*p2[0]*p3[0]
                                             + 2*p3[0]**2 + p1[1]*p2[1] - p1[1]*p3[1] - p2[1]*p3[1] + p3[1]**2)/(4*(self.nu**2 - 1)) / j_det
            K_local[1, 4] = K_local[4, 1] = (-self.nu*p1[0]*p2[1] + self.nu*p1[0]*p3[1] + 2*self.nu*p2[0]*p1[1] - self.nu*p2[0]*p2[1] - self.nu*p2[0] *
                                             p3[1] - 2*self.nu*p3[0]*p1[1] + 2*self.nu*p3[0]*p2[1] + p1[0]*p2[1] - p1[0]*p3[1] - p2[0]*p2[1] + p2[0]*p3[1])/(4*(self.nu**2 - 1)) / j_det
            K_local[1, 5] = K_local[5, 1] = (self.nu*p1[1]*p2[1] - self.nu*p1[1]*p3[1] - self.nu*p2[1]**2 + self.nu*p2[1]*p3[1] - 2*p1[0]*p2[0] +
                                             2*p1[0]*p3[0] + 2*p2[0]**2 - 2*p2[0]*p3[0] - p1[1]*p2[1] + p1[1]*p3[1] + p2[1]**2 - p2[1]*p3[1])/(4*(self.nu**2 - 1)) / j_det

            K_local[2, 2] = (self.nu*p1[0]**2 - 2*self.nu*p1[0]*p3[0] + self.nu*p3[0]**2 - p1[0]**2 + 2 *
                             p1[0]*p3[0] - p3[0]**2 - 2*p1[1]**2 + 4*p1[1]*p3[1] - 2*p3[1]**2)/(4*(self.nu**2 - 1)) / j_det
            K_local[2, 3] = K_local[3, 2] = (
                p1[0]*p1[1] - p1[0]*p3[1] - p3[0]*p1[1] + p3[0]*p3[1])/(4*(self.nu - 1)) / j_det
            K_local[2, 4] = K_local[4, 2] = (-self.nu*p1[0]**2 + self.nu*p1[0]*p2[0] + self.nu*p1[0]*p3[0] - self.nu*p2[0]*p3[0] + p1[0]**2 -
                                             p1[0]*p2[0] - p1[0]*p3[0] + p2[0]*p3[0] + 2*p1[1]**2 - 2*p1[1]*p2[1] - 2*p1[1]*p3[1] + 2*p2[1]*p3[1])/(4*(self.nu**2 - 1)) / j_det
            K_local[2, 5] = K_local[5, 2] = (-self.nu*p1[0]*p1[1] - self.nu*p1[0]*p2[1] + 2*self.nu*p1[0]*p3[1] + 2*self.nu*p2[0]*p1[1] - 2*self.nu*p2[0]
                                             * p3[1] - self.nu*p3[0]*p1[1] + self.nu*p3[0]*p2[1] - p1[0]*p1[1] + p1[0]*p2[1] + p3[0]*p1[1] - p3[0]*p2[1])/(4*(self.nu**2 - 1)) / j_det

            K_local[3, 3] = (self.nu*p1[1]**2 - 2*self.nu*p1[1]*p3[1] + self.nu*p3[1]**2 - 2*p1[0]**2 +
                             4*p1[0]*p3[0] - 2*p3[0]**2 - p1[1]**2 + 2*p1[1]*p3[1] - p3[1]**2)/(4*(self.nu**2 - 1)) / j_det
            K_local[3, 4] = K_local[4, 3] = (-self.nu*p1[0]*p1[1] + 2*self.nu*p1[0]*p2[1] - self.nu*p1[0]*p3[1] - self.nu*p2[0]*p1[1] + self.nu*p2[0] *
                                             p3[1] + 2*self.nu*p3[0]*p1[1] - 2*self.nu*p3[0]*p2[1] - p1[0]*p1[1] + p1[0]*p3[1] + p2[0]*p1[1] - p2[0]*p3[1])/(4*(self.nu**2 - 1)) / j_det
            K_local[3, 5] = K_local[5, 3] = (-self.nu*p1[1]**2 + self.nu*p1[1]*p2[1] + self.nu*p1[1]*p3[1] - self.nu*p2[1]*p3[1] + 2*p1[0]**2 -
                                             2*p1[0]*p2[0] - 2*p1[0]*p3[0] + 2*p2[0]*p3[0] + p1[1]**2 - p1[1]*p2[1] - p1[1]*p3[1] + p2[1]*p3[1])/(4*(self.nu**2 - 1)) / j_det

            K_local[4, 4] = (self.nu*p1[0]**2 - 2*self.nu*p1[0]*p2[0] + self.nu*p2[0]**2 - p1[0]**2 + 2 *
                             p1[0]*p2[0] - p2[0]**2 - 2*p1[1]**2 + 4*p1[1]*p2[1] - 2*p2[1]**2)/(4*(self.nu**2 - 1)) / j_det
            K_local[4, 5] = K_local[5, 4] = (
                p1[0]*p1[1] - p1[0]*p2[1] - p2[0]*p1[1] + p2[0]*p2[1])/(4*(self.nu - 1)) / j_det

            K_local[5, 5] = (self.nu*p1[1]**2 - 2*self.nu*p1[1]*p2[1] + self.nu*p2[1]**2 - 2*p1[0]**2 +
                             4*p1[0]*p2[0] - 2*p2[0]**2 - p1[1]**2 + 2*p1[1]*p2[1] - p2[1]**2)/(4*(self.nu**2 - 1)) / j_det
        else:
            K_local[0, 0] = (2.0*self.nu*p2[0]**2 - 4.0*self.nu*p2[0]*p3[0] + 2.0*self.nu*p3[0]**2 + 2.0*self.nu*p2[1]**2 - 4.0*self.nu*p2[1]*p3[1] + 2.0 *
                             self.nu*p3[1]**2 - p2[0]**2 + 2.0*p2[0]*p3[0] - p3[0]**2 - 2.0*p2[1]**2 + 4.0*p2[1]*p3[1] - 2.0*p3[1]**2)/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[0, 1] = K_local[1, 0] = (
                p2[0]*p2[1] - p2[0]*p3[1] - p3[0]*p2[1] + p3[0]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[0, 2] = K_local[2, 0] = (-2.0*self.nu*p1[0]*p2[0] + 2.0*self.nu*p1[0]*p3[0] + 2.0*self.nu*p2[0]*p3[0] - 2.0*self.nu*p3[0]**2 - 2.0*self.nu*p1[1]*p2[1] + 2.0*self.nu*p1[1]*p3[1] + 2.0*self.nu *
                                             p2[1]*p3[1] - 2.0*self.nu*p3[1]**2 + p1[0]*p2[0] - p1[0]*p3[0] - p2[0]*p3[0] + p3[0]**2 + 2.0*p1[1]*p2[1] - 2.0*p1[1]*p3[1] - 2.0*p2[1]*p3[1] + 2.0*p3[1]**2)/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[0, 3] = K_local[3, 0] = (-2.0*self.nu*p1[0]*p2[1] + 2.0*self.nu*p1[0]*p3[1] + 2.0*self.nu*p2[0]*p1[1] - 2.0*self.nu*p2[0]*p3[1] - 2.0 *
                                             self.nu*p3[0]*p1[1] + 2.0*self.nu*p3[0]*p2[1] - p2[0]*p1[1] + p2[0]*p3[1] + p3[0]*p1[1] - p3[0]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[0, 4] = K_local[4, 0] = (2.0*self.nu*p1[0]*p2[0] - 2.0*self.nu*p1[0]*p3[0] - 2.0*self.nu*p2[0]**2 + 2.0*self.nu*p2[0]*p3[0] + 2.0*self.nu*p1[1]*p2[1] - 2.0*self.nu*p1[1]*p3[1] - 2.0*self.nu *
                                             p2[1]**2 + 2.0*self.nu*p2[1]*p3[1] - p1[0]*p2[0] + p1[0]*p3[0] + p2[0]**2 - p2[0]*p3[0] - 2.0*p1[1]*p2[1] + 2.0*p1[1]*p3[1] + 2.0*p2[1]**2 - 2.0*p2[1]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[0, 5] = K_local[5, 0] = (2.0*self.nu*p1[0]*p2[1] - 2.0*self.nu*p1[0]*p3[1] - 2.0*self.nu*p2[0]*p1[1] + 2.0*self.nu*p2[0]*p3[1] + 2.0 *
                                             self.nu*p3[0]*p1[1] - 2.0*self.nu*p3[0]*p2[1] + p2[0]*p1[1] - p2[0]*p2[1] - p3[0]*p1[1] + p3[0]*p2[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)

            K_local[1, 1] = (2.0*self.nu*p2[0]**2 - 4.0*self.nu*p2[0]*p3[0] + 2.0*self.nu*p3[0]**2 + 2.0*self.nu*p2[1]**2 - 4.0*self.nu*p2[1]*p3[1] + 2.0 *
                             self.nu*p3[1]**2 - 2.0*p2[0]**2 + 4.0*p2[0]*p3[0] - 2.0*p3[0]**2 - p2[1]**2 + 2.0*p2[1]*p3[1] - p3[1]**2)/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[1, 2] = K_local[2, 1] = (2.0*self.nu*p1[0]*p2[1] - 2.0*self.nu*p1[0]*p3[1] - 2.0*self.nu*p2[0]*p1[1] + 2.0*self.nu*p2[0]*p3[1] + 2.0 *
                                             self.nu*p3[0]*p1[1] - 2.0*self.nu*p3[0]*p2[1] - p1[0]*p2[1] + p1[0]*p3[1] + p3[0]*p2[1] - p3[0]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[1, 3] = K_local[3, 1] = (-2.0*self.nu*p1[0]*p2[0] + 2.0*self.nu*p1[0]*p3[0] + 2.0*self.nu*p2[0]*p3[0] - 2.0*self.nu*p3[0]**2 - 2.0*self.nu*p1[1]*p2[1] + 2.0*self.nu*p1[1]*p3[1] + 2.0*self.nu *
                                             p2[1]*p3[1] - 2.0*self.nu*p3[1]**2 + 2.0*p1[0]*p2[0] - 2.0*p1[0]*p3[0] - 2.0*p2[0]*p3[0] + 2.0*p3[0]**2 + p1[1]*p2[1] - p1[1]*p3[1] - p2[1]*p3[1] + p3[1]**2)/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[1, 4] = K_local[4, 1] = (-2.0*self.nu*p1[0]*p2[1] + 2.0*self.nu*p1[0]*p3[1] + 2.0*self.nu*p2[0]*p1[1] - 2.0*self.nu*p2[0]*p3[1] - 2.0 *
                                             self.nu*p3[0]*p1[1] + 2.0*self.nu*p3[0]*p2[1] + p1[0]*p2[1] - p1[0]*p3[1] - p2[0]*p2[1] + p2[0]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[1, 5] = K_local[5, 1] = (2.0*self.nu*p1[0]*p2[0] - 2.0*self.nu*p1[0]*p3[0] - 2.0*self.nu*p2[0]**2 + 2.0*self.nu*p2[0]*p3[0] + 2.0*self.nu*p1[1]*p2[1] - 2.0*self.nu*p1[1]*p3[1] - 2.0*self.nu *
                                             p2[1]**2 + 2.0*self.nu*p2[1]*p3[1] - 2.0*p1[0]*p2[0] + 2.0*p1[0]*p3[0] + 2.0*p2[0]**2 - 2.0*p2[0]*p3[0] - p1[1]*p2[1] + p1[1]*p3[1] + p2[1]**2 - p2[1]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)

            K_local[2, 2] = (2.0*self.nu*p1[0]**2 - 4.0*self.nu*p1[0]*p3[0] + 2.0*self.nu*p3[0]**2 + 2.0*self.nu*p1[1]**2 - 4.0*self.nu*p1[1]*p3[1] + 2.0 *
                             self.nu*p3[1]**2 - p1[0]**2 + 2.0*p1[0]*p3[0] - p3[0]**2 - 2.0*p1[1]**2 + 4.0*p1[1]*p3[1] - 2.0*p3[1]**2)/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[2, 3] = K_local[3, 2] = (
                p1[0]*p1[1] - p1[0]*p3[1] - p3[0]*p1[1] + p3[0]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[2, 4] = K_local[4, 2] = (-2.0*self.nu*p1[0]**2 + 2.0*self.nu*p1[0]*p2[0] + 2.0*self.nu*p1[0]*p3[0] - 2.0*self.nu*p2[0]*p3[0] - 2.0*self.nu*p1[1]**2 + 2.0*self.nu*p1[1]*p2[1] + 2.0*self.nu *
                                             p1[1]*p3[1] - 2.0*self.nu*p2[1]*p3[1] + p1[0]**2 - p1[0]*p2[0] - p1[0]*p3[0] + p2[0]*p3[0] + 2.0*p1[1]**2 - 2.0*p1[1]*p2[1] - 2.0*p1[1]*p3[1] + 2.0*p2[1]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[2, 5] = K_local[5, 2] = (-2.0*self.nu*p1[0]*p2[1] + 2.0*self.nu*p1[0]*p3[1] + 2.0*self.nu*p2[0]*p1[1] - 2.0*self.nu*p2[0]*p3[1] - 2.0 *
                                             self.nu*p3[0]*p1[1] + 2.0*self.nu*p3[0]*p2[1] - p1[0]*p1[1] + p1[0]*p2[1] + p3[0]*p1[1] - p3[0]*p2[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)

            K_local[3, 3] = (2.0*self.nu*p1[0]**2 - 4.0*self.nu*p1[0]*p3[0] + 2.0*self.nu*p3[0]**2 + 2.0*self.nu*p1[1]**2 - 4.0*self.nu*p1[1]*p3[1] + 2.0 *
                             self.nu*p3[1]**2 - 2.0*p1[0]**2 + 4.0*p1[0]*p3[0] - 2.0*p3[0]**2 - p1[1]**2 + 2.0*p1[1]*p3[1] - p3[1]**2)/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[3, 4] = K_local[4, 3] = (2.0*self.nu*p1[0]*p2[1] - 2.0*self.nu*p1[0]*p3[1] - 2.0*self.nu*p2[0]*p1[1] + 2.0*self.nu*p2[0]*p3[1] + 2.0 *
                                             self.nu*p3[0]*p1[1] - 2.0*self.nu*p3[0]*p2[1] - p1[0]*p1[1] + p1[0]*p3[1] + p2[0]*p1[1] - p2[0]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[3, 5] = K_local[5, 3] = (-2.0*self.nu*p1[0]**2 + 2.0*self.nu*p1[0]*p2[0] + 2.0*self.nu*p1[0]*p3[0] - 2.0*self.nu*p2[0]*p3[0] - 2.0*self.nu*p1[1]**2 + 2.0*self.nu*p1[1]*p2[1] + 2.0*self.nu *
                                             p1[1]*p3[1] - 2.0*self.nu*p2[1]*p3[1] + 2.0*p1[0]**2 - 2.0*p1[0]*p2[0] - 2.0*p1[0]*p3[0] + 2.0*p2[0]*p3[0] + p1[1]**2 - p1[1]*p2[1] - p1[1]*p3[1] + p2[1]*p3[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)

            K_local[4, 4] = (2.0*self.nu*p1[0]**2 - 4.0*self.nu*p1[0]*p2[0] + 2.0*self.nu*p2[0]**2 + 2.0*self.nu*p1[1]**2 - 4.0*self.nu*p1[1]*p2[1] + 2.0 *
                             self.nu*p2[1]**2 - p1[0]**2 + 2.0*p1[0]*p2[0] - p2[0]**2 - 2.0*p1[1]**2 + 4.0*p1[1]*p2[1] - 2.0*p2[1]**2)/(8.0*self.nu**2 + 4.0*self.nu - 4.0)
            K_local[4, 5] = K_local[5, 4] = (
                p1[0]*p1[1] - p1[0]*p2[1] - p2[0]*p1[1] + p2[0]*p2[1])/(8.0*self.nu**2 + 4.0*self.nu - 4.0)

            K_local[5, 5] = (2.0*self.nu*p1[0]**2 - 4.0*self.nu*p1[0]*p2[0] + 2.0*self.nu*p2[0]**2 + 2.0*self.nu*p1[1]**2 - 4.0*self.nu*p1[1]*p2[1] + 2.0 *
                             self.nu*p2[1]**2 - 2.0*p1[0]**2 + 4.0*p1[0]*p2[0] - 2.0*p2[0]**2 - p1[1]**2 + 2.0*p1[1]*p2[1] - p2[1]**2)/(8.0*self.nu**2 + 4.0*self.nu - 4.0)

        # local mass matrix
        M_local = np.array([[2.0, 0, 1.0, 0, 1.0, 0], [0, 2.0, 0, 1.0, 0, 1.0], [1.0, 0, 2.0, 0, 1.0, 0], [
                           0, 1.0, 0, 2.0, 0, 1.0], [1.0, 0, 1.0, 0, 2.0, 0], [0, 1.0, 0, 1.0, 0, 2.0]]) * (j_det / 24.0)

        # b matrix
        b_local = j_det * self.gauss_quad.calculate(self.f, p1, p2, p3, dof=2)

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
            K_local, M_local, _ = self.calc_local_update(
                p1, p2, p3)

            # Assemble element's matrix solution into global matrix
            # columns = np.array([el_ps for _ in range(3)])
            plist = np.array([2*el_ps[0], 2*el_ps[0]+1, 2 *
                             el_ps[1], 2*el_ps[1]+1, 2*el_ps[2], 2*el_ps[2]+1])
            columns = np.array([plist for _ in range(6)])
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

            # Assemble element's matrix solution into global matrix
            # self.s[el_ps, 0] += b_local
            plist = np.array([2*el_ps[0], 2*el_ps[0]+1, 2 *
                             el_ps[1], 2*el_ps[1]+1, 2*el_ps[2], 2*el_ps[2]+1])
            self.s[plist, 0] += b_local

    def set_boundary_conditions_dirichlet(self):
        """
        Parameters
        ----------
        """
        # Set Dirichlet boundary conditions
        for key, value in self.mesh.bc_points_u["dirichlet"].items():
            self.u_dirichlet[2*key] = value
        for key, value in self.mesh.bc_points_v["dirichlet"].items():
            self.u_dirichlet[2*key+1] = value

    def set_boundary_conditions_neumann(self):
        """
        Parameters
        ----------
        """
        # Set Neumann boundary conditions
        for ch_idx, du_values in self.mesh.bc_points_u["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.tri.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.s[2*p_idx] += 0.5 * distance * du_value  # du_boundary

        for ch_idx, du_values in self.mesh.bc_points_v["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.tri.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.s[2*p_idx+1] += 0.5 * distance * du_value  # du_boundary

    def initialze(self):
        """
        Parameters
        ----------
        p1, p2, p3: numpy.ndarray
            coordinates of a triangle
        """
        # assign points to solve
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points_u["dirichlet"]:
                self.points_to_solve = np.append(
                    self.points_to_solve, 2*p_idx)

        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points_v["dirichlet"]:
                self.points_to_solve = np.append(
                    self.points_to_solve, 2*p_idx+1)

        if self.gpu:
            self.points_to_solve_d = cp.asarray(self.points_to_solve)

        # Calculate K and M entries
        self.set_K_M()

        # Evaluate source matrix
        self.set_s()

        # apply Neumann boundary conditions
        self.set_boundary_conditions_neumann()

        # apply dirichlet boundary conditions Neumann
        self.set_boundary_conditions_dirichlet()

        # Inverse mass matrix
        self.Minv = np.linalg.inv(self.M)

        # Calculate A and q entries
        self.A = self.Minv @ self.K
        self.b = self.Minv @ self.s - self.A @ self.u_dirichlet

        # Set the known
        for key, value in self.mesh.bc_points_u["dirichlet"].items():
            self.u[2*key] = value
        for key, value in self.mesh.bc_points_v["dirichlet"].items():
            self.u[2*key+1] = value

        # host to device data transfer
        if self.gpu:

            cp.cuda.runtime.memcpy(self.u_d.data.ptr, self.u.ctypes.data,
                                   self.u.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.u_dirichlet_d.data.ptr, self.u_dirichlet.ctypes.data,
                                   self.u_dirichlet.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            cp.cuda.runtime.memcpy(self.I_d.data.ptr, self.I.ctypes.data,
                                   self.I.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.A_d.data.ptr, self.A.ctypes.data,
                                   self.A.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.b_d.data.ptr, self.b.ctypes.data,
                                   self.b.nbytes, cp.cuda.runtime.memcpyHostToDevice)

    def solve(self):
        """
        Parameters
        ----------
        """
        # self.time_step_size()
        # dt_2 = self.dt

        # Solve u = A^-1 * b
        if not self.gpu:

            if self.sparse:
                K_sparse = sp.sparse.csr_matrix(
                    self.A[self.points_to_solve, :][:, self.points_to_solve])
                self.u[self.points_to_solve, 0], exitCode = sp.sparse.linalg.gmres(
                    K_sparse, self.b[self.points_to_solve], x0=self.u[self.points_to_solve, 0])
            else:
                self.u[self.points_to_solve] = sp.linalg.solve(
                    self.A[self.points_to_solve, :][:, self.points_to_solve], self.b[self.points_to_solve])
        else:

            if self.sparse:
                K_d_sparse = cps.sparse.csr_matrix(
                    self.A_d[self.points_to_solve_d, :][:, self.points_to_solve_d])
                self.u_d[self.points_to_solve_d, 0], exitCode = cps.sparse.linalg.gmres(
                    K_d_sparse, self.b_d[self.points_to_solve_d], x0=self.u_d[self.points_to_solve_d, 0])
            else:
                self.u_d[self.points_to_solve_d] = cp.linalg.solve(
                    self.A_d[self.points_to_solve_d, :][:, self.points_to_solve_d], self.b_d[self.points_to_solve_d])

            # device to host data transfer
            cp.cuda.runtime.memcpy(self.u.ctypes.data, self.u_d.data.ptr,
                                   self.u_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)

        # Set the known
        for key, value in self.mesh.bc_points_u["dirichlet"].items():
            self.u[2*key] = value
        for key, value in self.mesh.bc_points_v["dirichlet"].items():
            self.u[2*key+1] = value

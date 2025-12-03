# FEM_NS
import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy as cps
from cupyx.scipy.sparse.linalg import gmres
from FEM_tetrahedron_3d import GenericElement, GaussianQuadrature

# FEM Navier-Stokes 3D solver class


class FENS3D:
    """
    class for 2D Navier-Stokes equations solver with the finite element method

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
            Number of simplex
        n_points : int
            Number of points
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
        G1 : numpy.ndarray
            x-Gradient matrix
        G2 : numpy.ndarray
            y-Gradient matrix
        G3 : numpy.ndarray
            z-Gradient matrix
        E : numpy.ndarray
            Advection matrix
        A : numpy.ndarray
            A = Minv * K
        MG1 : numpy.ndarray
            MG1 = Minv * G1
        MG2 : numpy.ndarray
            MG2 = Minv * G2
        MG3 : numpy.ndarray
            MG3 = Minv * G3
        ME : numpy.ndarray
            ME = Minv * E
        s : numpy.ndarray
            Source vector for pressure
        su : numpy.ndarray
            source vector for x-velocity
        sv : numpy.ndarray
            source vector for y-velocity
        sw : numpy.ndarray
            source vector for z-velocity
        q : numpy.ndarray
            q = s - K * p_dirichlet
        qu : numpy.ndarray
            q = Minv * su
        qv : numpy.ndarray
            q = Minv * sv
        qw : numpy.ndarray
            q = Minv * sw
        b : numpy.ndarray
            RHS
        u : numpy.ndarray
            x-velocity
        v : numpy.ndarray
            y-velocity
        w : numpy.ndarray
            z-velocity
        p : numpy.ndarray
            pressure
        u_dirichlet : numpy.ndarray
            Dirichlet boundary x-velocity values
        v_dirichlet : numpy.ndarray
            Dirichlet boundary y-velocity values
        w_dirichlet : numpy.ndarray
            Dirichlet boundary z-velocity values
        p_dirichlet : numpy.ndarray
            Dirichlet boundary pressure values
        u_star : numpy.ndarray
            intermediate x-velocity
        v_star : numpy.ndarray
            intermediate y-velocity
        w_star : numpy.ndarray
            intermediate z-velocity
        points_to_solve : numpy.ndarray
            Index of points to solve for p
        points_to_solve_u : cupy.ndarray
            Index of points to solve for u
        points_to_solve_v : cupy.ndarray
            Index of points to solve for v
        points_to_solve_w : cupy.ndarray
            Index of points to solve for w
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
        G1_d : cupy.ndarray
            GPU G1 matrix
        G2_d : cupy.ndarray
            GPU G2 matrix
        G3_d : cupy.ndarray
            GPU G3 matrix
        MG1_d : cupy.ndarray
            GPU MG1 matrix
        MG2_d : cupy.ndarray
            GPU MG2 matrix
        MG3_d : cupy.ndarray
            GPU MG3 matrix
        ME_d : cupy.ndarray
            GPU ME matrix
        b_d : cupy.ndarray
            GPU b vector
        q_d : cupy.ndarray
            GPU q vector
        qu_d : cupy.ndarray
            GPU qu vector
        qv_d : cupy.ndarray
            GPU qv vector
        qw_d : cupy.ndarray
            GPU qw vector
        u_d : cupy.ndarray
            GPU x-velocity
        v_d : cupy.ndarray
            GPU y-velocity
        w_d : cupy.ndarray
            GPU z-velocity
        p_d : cupy.ndarray
            GPU pressure
        u_dirichlet_d : cupy.ndarray
            GPU u_dirichlet
        v_dirichlet_d : cupy.ndarray
            GPU v_dirichlet
        w_dirichlet_d : cupy.ndarray
            GPU w_dirichlet
        p_dirichlet_d : cupy.ndarray
            GPU p_dirichlet
        u_star_d : cupy.ndarray
            GPU u_star
        v_star_d : cupy.ndarray
            GPU v_star
        w_star_d : cupy.ndarray
            GPU w_star
        points_to_solve_d : cupy.ndarray
            GPU points_to_solve
        points_to_solve_u_d : cupy.ndarray
            GPU points_to_solve_u
        points_to_solve_v_d : cupy.ndarray
            GPU points_to_solve_v
        points_to_solve_w_d : cupy.ndarray
            GPU points_to_solve_w

    Methods
    -------
    time_step_size(self):
        Integration time step determination
    calc_local_update(self, p1, p2, p3, p4):
        Calculate the Jacobian, its determinant, and inverse
    set_K_M(self):
        Calculate the global mass and stiffness matrix
    set_E(self):
        Calculate the global advection matrix
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

    def __init__(self, _mesh, _f, _u=None, _v=None, _w=None, _p=None, _gpu=False, _sparse=False, _dt=0.001):
        """
        Parameters
        ----------
        _mesh : Mesh
            Mesh for computational domain
        _f : function
            R.H.S function
        _u : numpy.ndarray
            Initial guess for x-velocity
        _v : numpy.ndarray
            Initial guess for y-velocity
        _p : numpy.ndarray
            Initial guess for pressure
        _gpu : bool
            True: use GPU matrix solver, False: use CPU matrix solver, default CPU
        _sparse : bool
            True: use sparse matrix solver, False: use dense matrix solver, default Dense
        _dt : float
            Integration time step
        """
        self.gte = GenericElement()
        self.gauss_quad = GaussianQuadrature()

        self.mesh = _mesh
        self.n_elements = self.mesh.tri.nsimplex
        self.n_points = self.mesh.npoints

        self.f = _f

        self.M = np.zeros((self.n_points, self.n_points))
        self.Minv = np.zeros_like(self.M)
        self.K = np.zeros_like(self.M)
        self.A = np.zeros_like(self.M)
        self.G1 = np.zeros_like(self.M)
        self.G2 = np.zeros_like(self.M)
        self.G3 = np.zeros_like(self.M)
        self.E = np.zeros_like(self.M)
        self.MG1 = np.zeros_like(self.M)
        self.MG2 = np.zeros_like(self.M)
        self.MG3 = np.zeros_like(self.M)
        self.ME = np.zeros_like(self.M)
        self.I = np.eye(self.n_points)

        self.b = np.zeros((self.n_points, 1))
        self.su = np.zeros_like(self.b)
        self.sv = np.zeros_like(self.b)
        self.sw = np.zeros_like(self.b)
        self.s = np.zeros_like(self.b)
        self.qu = np.zeros_like(self.b)
        self.qv = np.zeros_like(self.b)
        self.qw = np.zeros_like(self.b)
        self.q = np.zeros_like(self.b)
        self.u_dirichlet = np.zeros_like(self.b)
        self.v_dirichlet = np.zeros_like(self.b)
        self.w_dirichlet = np.zeros_like(self.b)
        self.p_dirichlet = np.zeros_like(self.b)
        self.u_star = np.zeros_like(self.b)
        self.v_star = np.zeros_like(self.b)
        self.w_star = np.zeros_like(self.b)

        if _u is None:
            self.u = np.zeros_like(self.b)
        else:
            self.u = np.array(_u, copy=True)

        if _v is None:
            self.v = np.zeros_like(self.b)
        else:
            self.v = np.array(_v, copy=True)

        if _w is None:
            self.w = np.zeros_like(self.b)
        else:
            self.w = np.array(_w, copy=True)

        if _p is None:
            self.p = np.zeros_like(self.b)
        else:
            self.p = np.array(_p, copy=True)

        self.dt = _dt
        self.nu = 1

        self.points_to_solve = np.array([], dtype=np.int32)
        self.points_to_solve_u = np.array([], dtype=np.int32)
        self.points_to_solve_v = np.array([], dtype=np.int32)
        self.points_to_solve_w = np.array([], dtype=np.int32)

        self.gpu = _gpu
        self.sparse = _sparse
        if self.gpu:
            # Memory Pools for Efficient Allocation
            self.mp = cp.get_default_memory_pool()
            self.pp = cp.get_default_pinned_memory_pool()

            self.K_d = cp.zeros((self.n_points, self.n_points))
            self.I_d = cp.zeros_like(self.K_d)
            self.A_d = cp.zeros_like(self.K_d)

            self.G1_d = cp.zeros_like(self.K_d)
            self.G2_d = cp.zeros_like(self.K_d)
            self.G3_d = cp.zeros_like(self.K_d)
            self.MG1_d = cp.zeros_like(self.K_d)
            self.MG2_d = cp.zeros_like(self.K_d)
            self.MG3_d = cp.zeros_like(self.K_d)
            self.ME_d = cp.zeros_like(self.K_d)

            self.b_d = cp.zeros((self.n_points, 1))
            self.qu_d = cp.zeros_like(self.b_d)
            self.qv_d = cp.zeros_like(self.b_d)
            self.qw_d = cp.zeros_like(self.b_d)
            self.q_d = cp.zeros_like(self.b_d)

            self.u_d = cp.zeros_like(self.b_d)
            self.v_d = cp.zeros_like(self.b_d)
            self.w_d = cp.zeros_like(self.b_d)
            self.p_d = cp.zeros_like(self.b_d)

            self.u_star_d = cp.zeros_like(self.b_d)
            self.v_star_d = cp.zeros_like(self.b_d)
            self.w_star_d = cp.zeros_like(self.b_d)

            self.u_dirichlet_d = cp.zeros_like(self.b_d)
            self.v_dirichlet_d = cp.zeros_like(self.b_d)
            self.w_dirichlet_d = cp.zeros_like(self.b_d)
            self.p_dirichlet_d = cp.zeros_like(self.b_d)

            self.points_to_solve_d = cp.array([], dtype=cp.int32)
            self.points_to_solve_u_d = cp.array([], dtype=cp.int32)
            self.points_to_solve_v_d = cp.array([], dtype=cp.int32)
            self.points_to_solve_w_d = cp.array([], dtype=cp.int32)

        print('Solving using GPU:', self.gpu)
        print('Solving using sparse matrix:', self.sparse)

    def time_step_size(self):
        """
        Parameters
        ----------
        """
        # time step determination
        dmin2 = 1.0
        CFL = 0.5
        for i, el_ps in enumerate(self.mesh.tri.simplices):
            p1, p2, p3, p4 = (self.mesh.tri.points[el_ps[0]],
                              self.mesh.tri.points[el_ps[1]],
                              self.mesh.tri.points[el_ps[2]],
                              self.mesh.tri.points[el_ps[3]])

            d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2
            d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2 + (p1[2]-p3[2])**2
            d14 = (p1[0]-p4[0])**2 + (p1[1]-p4[1])**2 + (p1[2]-p4[2])**2
            d32 = (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2 + (p3[2]-p2[2])**2
            d42 = (p4[0]-p2[0])**2 + (p4[1]-p2[1])**2 + (p4[2]-p2[2])**2
            d34 = (p3[0]-p4[0])**2 + (p3[1]-p4[1])**2 + (p3[2]-p4[2])**2
            dmin2 = min(dmin2, d12, d13, d14, d32, d42, d34)

        vel_max = max(np.abs(self.u).max(), np.abs(
            self.v).max(), np.abs(self.w).max()) + np.sqrt(1.4 * self.p.max())
        dt_conv = np.sqrt(dmin2) * CFL / vel_max
        dt_diff = dmin2
        self.dt = dt_conv  # min(dt_diff, dt_conv)

    # @staticmethod
    def calc_local_update(self, p1, p2, p3, p4):
        """
        Parameters
        ----------
        p1, p2, p3, p4: numpy.ndarray
            Coordinates of a triangle
        """
        # Calculate the Jacobian, its determinant, and inverse
        j = np.array([[p1[0] - p4[0], p1[1] - p4[1], p1[2] - p4[2]],
                      [p2[0] - p4[0], p2[1] - p4[1], p2[2] - p4[2]],
                      [p3[0] - p4[0], p3[1] - p4[1], p3[2] - p4[2]]])
        j_det = np.linalg.det(j)
        j_inv = np.linalg.inv(j)

        # Calculate matrix solution of one element
        K_local = np.zeros((4, 4))
        G1_local = np.zeros_like(K_local)
        G2_local = np.zeros_like(K_local)
        G3_local = np.zeros_like(K_local)
        b_local = np.zeros((4, 1))

        # local stiffness matrix
        K_local[0, 0] = ((p2[0]*p3[1] - p2[0]*p4[1] - p3[0]*p2[1] + p3[0]*p4[1] + p4[0]*p2[1] - p4[0]*p3[1])**2 + (p2[0]*p3[2] - p2[0]*p4[2] - p3[0]*p2[2] + p3[0]*p4[2] + p4[0]*p2[2] - p4[0]*p3[2])**2 + (p2[1]*p3[2] - p2[1]*p4[2] - p3[1]*p2[2] + p3[1]*p4[2] + p4[1]*p2[2] - p4[1]*p3[2])**2)/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] -
                                                                                                                                                                                                                                                                                                    p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[0, 1] = K_local[1, 0] = (-(p1[0]*p3[1] - p1[0]*p4[1] - p3[0]*p1[1] + p3[0]*p4[1] + p4[0]*p1[1] - p4[0]*p3[1])*(p2[0]*p3[1] - p2[0]*p4[1] - p3[0]*p2[1] + p3[0]*p4[1] + p4[0]*p2[1] - p4[0]*p3[1]) - (p1[0]*p3[2] - p1[0]*p4[2] - p3[0]*p1[2] + p3[0]*p4[2] + p4[0]*p1[2] - p4[0]*p3[2])*(p2[0]*p3[2] - p2[0]*p4[2] - p3[0]*p2[2] + p3[0]*p4[2] + p4[0]*p2[2] - p4[0]*p3[2]) - (p1[1]*p3[2] - p1[1]*p4[2] - p3[1]*p1[2] + p3[1]*p4[2] + p4[1]*p1[2] - p4[1]*p3[2])*(p2[1]*p3[2] - p2[1]*p4[2] - p3[1]*p2[2] + p3[1]*p4[2] +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   p4[1]*p2[2] - p4[1]*p3[2]))/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] - p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[0, 2] = K_local[2, 0] = ((p1[0]*p2[1] - p1[0]*p4[1] - p2[0]*p1[1] + p2[0]*p4[1] + p4[0]*p1[1] - p4[0]*p2[1])*(p2[0]*p3[1] - p2[0]*p4[1] - p3[0]*p2[1] + p3[0]*p4[1] + p4[0]*p2[1] - p4[0]*p3[1]) + (p1[0]*p2[2] - p1[0]*p4[2] - p2[0]*p1[2] + p2[0]*p4[2] + p4[0]*p1[2] - p4[0]*p2[2])*(p2[0]*p3[2] - p2[0]*p4[2] - p3[0]*p2[2] + p3[0]*p4[2] + p4[0]*p2[2] - p4[0]*p3[2]) + (p1[1]*p2[2] - p1[1]*p4[2] - p2[1]*p1[2] + p2[1]*p4[2] + p4[1]*p1[2] - p4[1]*p2[2])*(p2[1]*p3[2] - p2[1]*p4[2] - p3[1]*p2[2] + p3[1]*p4[2] +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  p4[1]*p2[2] - p4[1]*p3[2]))/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] - p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[0, 3] = K_local[3, 0] = (-(p1[0]*p2[1] - p1[0]*p3[1] - p2[0]*p1[1] + p2[0]*p3[1] + p3[0]*p1[1] - p3[0]*p2[1])*(p2[0]*p3[1] - p2[0]*p4[1] - p3[0]*p2[1] + p3[0]*p4[1] + p4[0]*p2[1] - p4[0]*p3[1]) - (p1[0]*p2[2] - p1[0]*p3[2] - p2[0]*p1[2] + p2[0]*p3[2] + p3[0]*p1[2] - p3[0]*p2[2])*(p2[0]*p3[2] - p2[0]*p4[2] - p3[0]*p2[2] + p3[0]*p4[2] + p4[0]*p2[2] - p4[0]*p3[2]) - (p1[1]*p2[2] - p1[1]*p3[2] - p2[1]*p1[2] + p2[1]*p3[2] + p3[1]*p1[2] - p3[1]*p2[2])*(p2[1]*p3[2] - p2[1]*p4[2] - p3[1]*p2[2] + p3[1]*p4[2] +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   p4[1]*p2[2] - p4[1]*p3[2]))/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] - p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[1, 1] = ((p1[0]*p3[1] - p1[0]*p4[1] - p3[0]*p1[1] + p3[0]*p4[1] + p4[0]*p1[1] - p4[0]*p3[1])**2 + (p1[0]*p3[2] - p1[0]*p4[2] - p3[0]*p1[2] + p3[0]*p4[2] + p4[0]*p1[2] - p4[0]*p3[2])**2 + (p1[1]*p3[2] - p1[1]*p4[2] - p3[1]*p1[2] + p3[1]*p4[2] + p4[1]*p1[2] - p4[1]*p3[2])**2)/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] -
                                                                                                                                                                                                                                                                                                    p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[1, 2] = K_local[2, 1] = (-(p1[0]*p2[1] - p1[0]*p4[1] - p2[0]*p1[1] + p2[0]*p4[1] + p4[0]*p1[1] - p4[0]*p2[1])*(p1[0]*p3[1] - p1[0]*p4[1] - p3[0]*p1[1] + p3[0]*p4[1] + p4[0]*p1[1] - p4[0]*p3[1]) - (p1[0]*p2[2] - p1[0]*p4[2] - p2[0]*p1[2] + p2[0]*p4[2] + p4[0]*p1[2] - p4[0]*p2[2])*(p1[0]*p3[2] - p1[0]*p4[2] - p3[0]*p1[2] + p3[0]*p4[2] + p4[0]*p1[2] - p4[0]*p3[2]) - (p1[1]*p2[2] - p1[1]*p4[2] - p2[1]*p1[2] + p2[1]*p4[2] + p4[1]*p1[2] - p4[1]*p2[2])*(p1[1]*p3[2] - p1[1]*p4[2] - p3[1]*p1[2] + p3[1]*p4[2] +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   p4[1]*p1[2] - p4[1]*p3[2]))/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] - p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[1, 3] = K_local[3, 1] = ((p1[0]*p2[1] - p1[0]*p3[1] - p2[0]*p1[1] + p2[0]*p3[1] + p3[0]*p1[1] - p3[0]*p2[1])*(p1[0]*p3[1] - p1[0]*p4[1] - p3[0]*p1[1] + p3[0]*p4[1] + p4[0]*p1[1] - p4[0]*p3[1]) + (p1[0]*p2[2] - p1[0]*p3[2] - p2[0]*p1[2] + p2[0]*p3[2] + p3[0]*p1[2] - p3[0]*p2[2])*(p1[0]*p3[2] - p1[0]*p4[2] - p3[0]*p1[2] + p3[0]*p4[2] + p4[0]*p1[2] - p4[0]*p3[2]) + (p1[1]*p2[2] - p1[1]*p3[2] - p2[1]*p1[2] + p2[1]*p3[2] + p3[1]*p1[2] - p3[1]*p2[2])*(p1[1]*p3[2] - p1[1]*p4[2] - p3[1]*p1[2] + p3[1]*p4[2] +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  p4[1]*p1[2] - p4[1]*p3[2]))/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] - p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[2, 2] = ((p1[0]*p2[1] - p1[0]*p4[1] - p2[0]*p1[1] + p2[0]*p4[1] + p4[0]*p1[1] - p4[0]*p2[1])**2 + (p1[0]*p2[2] - p1[0]*p4[2] - p2[0]*p1[2] + p2[0]*p4[2] + p4[0]*p1[2] - p4[0]*p2[2])**2 + (p1[1]*p2[2] - p1[1]*p4[2] - p2[1]*p1[2] + p2[1]*p4[2] + p4[1]*p1[2] - p4[1]*p2[2])**2)/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] -
                                                                                                                                                                                                                                                                                                    p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[2, 3] = K_local[3, 2] = (-(p1[0]*p2[1] - p1[0]*p3[1] - p2[0]*p1[1] + p2[0]*p3[1] + p3[0]*p1[1] - p3[0]*p2[1])*(p1[0]*p2[1] - p1[0]*p4[1] - p2[0]*p1[1] + p2[0]*p4[1] + p4[0]*p1[1] - p4[0]*p2[1]) - (p1[0]*p2[2] - p1[0]*p3[2] - p2[0]*p1[2] + p2[0]*p3[2] + p3[0]*p1[2] - p3[0]*p2[2])*(p1[0]*p2[2] - p1[0]*p4[2] - p2[0]*p1[2] + p2[0]*p4[2] + p4[0]*p1[2] - p4[0]*p2[2]) - (p1[1]*p2[2] - p1[1]*p3[2] - p2[1]*p1[2] + p2[1]*p3[2] + p3[1]*p1[2] - p3[1]*p2[2])*(p1[1]*p2[2] - p1[1]*p4[2] - p2[1]*p1[2] + p2[1]*p4[2] +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   p4[1]*p1[2] - p4[1]*p2[2]))/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] - p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        K_local[3, 3] = ((p1[0]*p2[1] - p1[0]*p3[1] - p2[0]*p1[1] + p2[0]*p3[1] + p3[0]*p1[1] - p3[0]*p2[1])**2 + (p1[0]*p2[2] - p1[0]*p3[2] - p2[0]*p1[2] + p2[0]*p3[2] + p3[0]*p1[2] - p3[0]*p2[2])**2 + (p1[1]*p2[2] - p1[1]*p3[2] - p2[1]*p1[2] + p2[1]*p3[2] + p3[1]*p1[2] - p3[1]*p2[2])**2)/(p1[0]*p2[1]*p3[2] - p1[0]*p2[1]*p4[2] - p1[0]*p3[1]*p2[2] + p1[0]*p3[1]*p4[2] + p1[0]*p4[1]*p2[2] -
                                                                                                                                                                                                                                                                                                    p1[0]*p4[1]*p3[2] - p2[0]*p1[1]*p3[2] + p2[0]*p1[1]*p4[2] + p2[0]*p3[1]*p1[2] - p2[0]*p3[1]*p4[2] - p2[0]*p4[1]*p1[2] + p2[0]*p4[1]*p3[2] + p3[0]*p1[1]*p2[2] - p3[0]*p1[1]*p4[2] - p3[0]*p2[1]*p1[2] + p3[0]*p2[1]*p4[2] + p3[0]*p4[1]*p1[2] - p3[0]*p4[1]*p2[2] - p4[0]*p1[1]*p2[2] + p4[0]*p1[1]*p3[2] + p4[0]*p2[1]*p1[2] - p4[0]*p2[1]*p3[2] - p4[0]*p3[1]*p1[2] + p4[0]*p3[1]*p2[2])**2

        # local mass matrix
        M_local = np.array([[2.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0], [
                           1.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 2.0]]) * (j_det / 120.0)

        # local G and H matrix
        G1_local[0, 0] = G1_local[1, 0] = G1_local[2,
                                                   0] = G1_local[3, 0] = j_inv[0, 0] * j_det / 24.0
        G1_local[0, 1] = G1_local[1, 1] = G1_local[2,
                                                   1] = G1_local[3, 1] = j_inv[0, 1] * j_det / 24.0
        G1_local[0, 2] = G1_local[1, 2] = G1_local[2,
                                                   2] = G1_local[3, 2] = j_inv[0, 2] * j_det / 24.0
        G1_local[0, 3] = G1_local[1, 3] = G1_local[2, 3] = G1_local[3,
                                                                    3] = -(j_inv[0, 0] + j_inv[0, 1] + j_inv[0, 2]) * j_det / 24.0

        G2_local[0, 0] = G2_local[1, 0] = G2_local[2,
                                                   0] = G2_local[3, 0] = j_inv[1, 0] * j_det / 24.0
        G2_local[0, 1] = G2_local[1, 1] = G2_local[2,
                                                   1] = G2_local[3, 1] = j_inv[1, 1] * j_det / 24.0
        G2_local[0, 2] = G2_local[1, 2] = G2_local[2,
                                                   2] = G2_local[3, 2] = j_inv[1, 2] * j_det / 24.0
        G2_local[0, 3] = G2_local[1, 3] = G2_local[2, 3] = G2_local[3,
                                                                    3] = -(j_inv[1, 0] + j_inv[1, 1] + j_inv[1, 2]) * j_det / 24.0

        G3_local[0, 0] = G3_local[1, 0] = G3_local[2,
                                                   0] = G3_local[3, 0] = j_inv[2, 0] * j_det / 24.0
        G3_local[0, 1] = G3_local[1, 1] = G3_local[2,
                                                   1] = G3_local[3, 1] = j_inv[2, 1] * j_det / 24.0
        G3_local[0, 2] = G3_local[1, 2] = G3_local[2,
                                                   2] = G3_local[3, 2] = j_inv[2, 2] * j_det / 24.0
        G3_local[0, 3] = G3_local[1, 3] = G3_local[2, 3] = G3_local[3,
                                                                    3] = -(j_inv[2, 0] + j_inv[2, 1] + j_inv[2, 2]) * j_det / 24.0

        # b matrix
        # b_local = j_det * self.gauss_quad.calculate(self.f, p1, p2, p3, p4)

        return K_local * j_det / 6.0, M_local, b_local, G1_local, G2_local, G3_local

    def set_K_M(self):
        """
        Parameters
        ----------
        """
        # Calculate the global mass and stiffness matrix
        for i, el_ps in enumerate(self.mesh.tri.simplices):
            # Extract element's nodes
            p1, p2, p3, p4 = (self.mesh.tri.points[el_ps[0]],
                              self.mesh.tri.points[el_ps[1]],
                              self.mesh.tri.points[el_ps[2]],
                              self.mesh.tri.points[el_ps[3]])

            # Store local element's solution
            K_local, M_local, _, G1_local, G2_local, G3_local = self.calc_local_update(
                p1, p2, p3, p4)

            # Assemble element's matrix solution into global matrix
            columns = np.array([self.mesh.pmap[el_ps] for _ in range(4)])
            rows = columns.T
            self.K[rows, columns] += K_local
            self.M[rows, columns] += M_local
            self.G1[rows, columns] += G1_local
            self.G2[rows, columns] += G2_local
            self.G3[rows, columns] += G3_local

    def set_E(self):
        """
        Parameters
        ----------
        """
        # Calculate the global source vector
        for i, el_ps in enumerate(self.mesh.tri.simplices):
            # Extract element's nodes
            p1, p2, p3, p4 = (self.mesh.tri.points[el_ps[0]],
                              self.mesh.tri.points[el_ps[1]],
                              self.mesh.tri.points[el_ps[2]],
                              self.mesh.tri.points[el_ps[3]])

            # Store local element's solution
            yzbar = np.zeros((1, 4))
            xzbar = np.zeros((1, 4))
            xybar = np.zeros((1, 4))
            ubar = np.zeros((4, 1))
            vbar = np.zeros((4, 1))
            wbar = np.zeros((4, 1))

            yzbar[0, :] = (1.0/120.0) * np.array([p2[1]*p3[2] - p2[1]*p4[2] - p3[1]*p2[2] + p3[1]*p4[2] + p4[1]*p2[2] - p4[1]*p3[2],
                                                  -p1[1]*p3[2] + p1[1]*p4[2] + p3[1]*p1[2] -
                                                  p3[1]*p4[2] - p4[1] *
                                                  p1[2] + p4[1]*p3[2],
                                                  p1[1]*p2[2] - p1[1]*p4[2] - p2[1]*p1[2] +
                                                  p2[1]*p4[2] + p4[1] *
                                                  p1[2] - p4[1]*p2[2],
                                                  -p1[1]*p2[2] + p1[1]*p3[2] + p2[1]*p1[2] - p2[1]*p3[2] - p3[1]*p1[2] + p3[1]*p2[2]])

            xzbar[0, :] = (1.0/120.0) * np.array([-p2[0]*p3[2] + p2[0]*p4[2] + p3[0]*p2[2] - p3[0]*p4[2] - p4[0]*p2[2] + p4[0]*p3[2],
                                                  p1[0]*p3[2] - p1[0]*p4[2] - p3[0]*p1[2] +
                                                  p3[0]*p4[2] + p4[0] *
                                                  p1[2] - p4[0]*p3[2],
                                                  -p1[0]*p2[2] + p1[0]*p4[2] + p2[0]*p1[2] -
                                                  p2[0]*p4[2] - p4[0] *
                                                  p1[2] + p4[0]*p2[2],
                                                  p1[0]*p2[2] - p1[0]*p3[2] - p2[0]*p1[2] + p2[0]*p3[2] + p3[0]*p1[2] - p3[0]*p2[2]])

            xybar[0, :] = (1.0/120.0) * np.array([p2[0]*p3[1] - p2[0]*p4[1] - p3[0]*p2[1] + p3[0]*p4[1] + p4[0]*p2[1] - p4[0]*p3[1],
                                                  -p1[0]*p3[1] + p1[0]*p4[1] + p3[0]*p1[1] -
                                                  p3[0]*p4[1] - p4[0] *
                                                  p1[1] + p4[0]*p3[1],
                                                  p1[0]*p2[1] - p1[0]*p4[1] - p2[0]*p1[1] +
                                                  p2[0]*p4[1] + p4[0] *
                                                  p1[1] - p4[0]*p2[1],
                                                  -p1[0]*p2[1] + p1[0]*p3[1] + p2[0]*p1[1] - p2[0]*p3[1] - p3[0]*p1[1] + p3[0]*p2[1]])

            ubar[:, 0] = np.array([2*self.u[self.mesh.pmap[el_ps[0]], 0] + self.u[self.mesh.pmap[el_ps[1]], 0] + self.u[self.mesh.pmap[el_ps[2]], 0] + self.u[self.mesh.pmap[el_ps[3]], 0],
                                   self.u[self.mesh.pmap[el_ps[0]], 0] + 2*self.u[self.mesh.pmap[el_ps[1]], 0] +
                                   self.u[self.mesh.pmap[el_ps[2]], 0] +
                                   self.u[self.mesh.pmap[el_ps[3]], 0],
                                   self.u[self.mesh.pmap[el_ps[0]], 0] + self.u[self.mesh.pmap[el_ps[1]], 0] +
                                   2*self.u[self.mesh.pmap[el_ps[2]], 0] +
                                   self.u[self.mesh.pmap[el_ps[3]], 0],
                                   self.u[self.mesh.pmap[el_ps[0]], 0] + self.u[self.mesh.pmap[el_ps[1]], 0] +
                                   self.u[self.mesh.pmap[el_ps[2]], 0] +
                                   2*self.u[self.mesh.pmap[el_ps[3]], 0],
                                   ])

            vbar[:, 0] = np.array([2*self.v[self.mesh.pmap[el_ps[0]], 0] + self.v[self.mesh.pmap[el_ps[1]], 0] + self.v[self.mesh.pmap[el_ps[2]], 0] + self.v[self.mesh.pmap[el_ps[3]], 0],
                                   self.v[self.mesh.pmap[el_ps[0]], 0] + 2*self.v[self.mesh.pmap[el_ps[1]], 0] +
                                   self.v[self.mesh.pmap[el_ps[2]], 0] +
                                   self.v[self.mesh.pmap[el_ps[3]], 0],
                                   self.v[self.mesh.pmap[el_ps[0]], 0] + self.v[self.mesh.pmap[el_ps[1]], 0] +
                                   2*self.v[self.mesh.pmap[el_ps[2]], 0] +
                                   self.v[self.mesh.pmap[el_ps[3]], 0],
                                   self.v[self.mesh.pmap[el_ps[0]], 0] + self.v[self.mesh.pmap[el_ps[1]], 0] +
                                   self.v[self.mesh.pmap[el_ps[2]], 0] +
                                   2*self.v[self.mesh.pmap[el_ps[3]], 0]
                                   ])

            wbar[:, 0] = np.array([2*self.w[self.mesh.pmap[el_ps[0]], 0] + self.w[self.mesh.pmap[el_ps[1]], 0] + self.w[self.mesh.pmap[el_ps[2]], 0] + self.w[self.mesh.pmap[el_ps[3]], 0],
                                   self.w[self.mesh.pmap[el_ps[0]], 0] + 2*self.w[self.mesh.pmap[el_ps[1]], 0] +
                                   self.w[self.mesh.pmap[el_ps[2]], 0] +
                                   self.w[self.mesh.pmap[el_ps[3]], 0],
                                   self.w[self.mesh.pmap[el_ps[0]], 0] + self.w[self.mesh.pmap[el_ps[1]], 0] +
                                   2*self.w[self.mesh.pmap[el_ps[2]], 0] +
                                   self.w[self.mesh.pmap[el_ps[3]], 0],
                                   self.w[self.mesh.pmap[el_ps[0]], 0] + self.w[self.mesh.pmap[el_ps[1]], 0] +
                                   self.w[self.mesh.pmap[el_ps[2]], 0] +
                                   2*self.w[self.mesh.pmap[el_ps[3]], 0]
                                   ])

            E_local = ubar @ yzbar + vbar @ xzbar + wbar @ xybar

            # Assemble element's matrix solution into global matrix
            columns = np.array([self.mesh.pmap[el_ps] for _ in range(4)])
            rows = columns.T
            self.E[rows, columns] += E_local

    def set_s(self):
        """
        Parameters
        ----------
        """
        # Calculate the global source vector
        for i, el_ps in enumerate(self.mesh.tri.simplices):
            # Extract element's nodes
            p1, p2, p3, p4 = (self.mesh.tri.points[el_ps[0]],
                              self.mesh.tri.points[el_ps[1]],
                              self.mesh.tri.points[el_ps[2]],
                              self.mesh.tri.points[el_ps[3]])

            # Store local element's solution
            _, _, b_local, _, _, _ = self.calc_local_update(p1, p2, p3, p4)

            # Assemble element's matrix solution into global matrix
            self.s[self.mesh.pmap[el_ps], 0] += b_local[:, 0]

    def set_boundary_conditions_dirichlet(self):
        """
        Parameters
        ----------
        """
        # Set Dirichlet boundary conditions
        for key, value in self.mesh.bc_points_u["dirichlet"].items():
            self.u_dirichlet[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_v["dirichlet"].items():
            self.v_dirichlet[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_w["dirichlet"].items():
            self.w_dirichlet[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points["dirichlet"].items():
            self.p_dirichlet[self.mesh.pmap[key]] = value

    def set_boundary_conditions_neumann(self):
        """
        Parameters
        ----------
        """
        # Set Neumann boundary conditions
        for ch_idx, du_values in self.mesh.bc_points_u["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt(((p1 - p2)**2).sum())
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.su[self.mesh.pmap[p_idx]] += 0.5 * \
                    distance * du_value  # du_boundary

        for ch_idx, du_values in self.mesh.bc_points_v["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt(((p1 - p2)**2).sum())
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.sv[self.mesh.pmap[p_idx]] += 0.5 * \
                    distance * du_value  # du_boundary

        for ch_idx, du_values in self.mesh.bc_points_w["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt(((p1 - p2)**2).sum())
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.sw[self.mesh.pmap[p_idx]] += 0.5 * \
                    distance * du_value  # du_boundary

        for ch_idx, du_values in self.mesh.bc_points["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt(((p1 - p2)**2).sum())
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.s[self.mesh.pmap[p_idx]] += 0.5 * \
                    distance * du_value  # du_boundary

    def initialize(self):
        """
        Parameters
        ----------
        """
        # assign points to solve
        counter = 0
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points["dirichlet"] and self.mesh.pflg[p_idx]:
                self.points_to_solve = np.append(
                    self.points_to_solve, self.mesh.pmap[p_idx])

        # assign points to solve
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points_u["dirichlet"] and self.mesh.pflg[p_idx]:
                self.points_to_solve_u = np.append(
                    self.points_to_solve_u, self.mesh.pmap[p_idx])

        # assign points to solve
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points_v["dirichlet"] and self.mesh.pflg[p_idx]:
                self.points_to_solve_v = np.append(
                    self.points_to_solve_v, self.mesh.pmap[p_idx])

        # assign points to solve
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points_w["dirichlet"] and self.mesh.pflg[p_idx]:
                self.points_to_solve_w = np.append(
                    self.points_to_solve_w, self.mesh.pmap[p_idx])

        if self.gpu:
            self.points_to_solve_d = cp.asarray(self.points_to_solve)
            self.points_to_solve_u_d = cp.asarray(self.points_to_solve_u)
            self.points_to_solve_v_d = cp.asarray(self.points_to_solve_v)
            self.points_to_solve_w_d = cp.asarray(self.points_to_solve_w)

        # Calculate K and M entries
        self.set_K_M()

        # Inverse mass matrix
        self.Minv = np.linalg.inv(self.M)

        # Evaluate source matrix
        self.set_s()

        # apply Neumann boundary conditions
        self.set_boundary_conditions_neumann()

        # apply dirichlet boundary conditions Neumann
        self.set_boundary_conditions_dirichlet()

        # Calculate q entries
        self.qu = self.Minv @ self.su * self.nu
        self.qv = self.Minv @ self.sv * self.nu
        self.qw = self.Minv @ self.sw * self.nu
        self.q = self.s - self.K @ self.p_dirichlet

        self.A = self.Minv @ self.K * self.nu
        self.MG1 = self.Minv @ self.G1
        self.MG2 = self.Minv @ self.G2
        self.MG3 = self.Minv @ self.G3

        # Set the known
        for key, value in self.mesh.bc_points_u["dirichlet"].items():
            self.u[self.mesh.pmap[key]] = value
            self.u_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_v["dirichlet"].items():
            self.v[self.mesh.pmap[key]] = value
            self.v_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_w["dirichlet"].items():
            self.w[self.mesh.pmap[key]] = value
            self.w_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points["dirichlet"].items():
            self.p[self.mesh.pmap[key]] = value

        # host to device data transfer
        if self.gpu:

            cp.cuda.runtime.memcpy(self.u_d.data.ptr, self.u.ctypes.data,
                                   self.u.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.v_d.data.ptr, self.v.ctypes.data,
                                   self.v.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.w_d.data.ptr, self.w.ctypes.data,
                                   self.w.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.p_d.data.ptr, self.p.ctypes.data,
                                   self.p.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            cp.cuda.runtime.memcpy(self.u_star_d.data.ptr, self.u_star.ctypes.data,
                                   self.u_star.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.v_star_d.data.ptr, self.v_star.ctypes.data,
                                   self.v_star.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.w_star_d.data.ptr, self.w_star.ctypes.data,
                                   self.w_star.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            cp.cuda.runtime.memcpy(self.u_dirichlet_d.data.ptr, self.u_dirichlet.ctypes.data,
                                   self.u_dirichlet.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.v_dirichlet_d.data.ptr, self.v_dirichlet.ctypes.data,
                                   self.v_dirichlet.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.w_dirichlet_d.data.ptr, self.w_dirichlet.ctypes.data,
                                   self.w_dirichlet.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            cp.cuda.runtime.memcpy(self.I_d.data.ptr, self.I.ctypes.data,
                                   self.I.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.K_d.data.ptr, self.K.ctypes.data,
                                   self.K.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.A_d.data.ptr, self.A.ctypes.data,
                                   self.A.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            cp.cuda.runtime.memcpy(self.G1_d.data.ptr, self.G1.ctypes.data,
                                   self.G1.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.G2_d.data.ptr, self.G2.ctypes.data,
                                   self.G2.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.G3_d.data.ptr, self.G3.ctypes.data,
                                   self.G3.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.MG1_d.data.ptr, self.MG1.ctypes.data,
                                   self.MG1.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.MG2_d.data.ptr, self.MG2.ctypes.data,
                                   self.MG2.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.MG3_d.data.ptr, self.MG3.ctypes.data,
                                   self.MG3.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            cp.cuda.runtime.memcpy(self.qu_d.data.ptr, self.qu.ctypes.data,
                                   self.qu.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.qv_d.data.ptr, self.qv.ctypes.data,
                                   self.qv.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.qw_d.data.ptr, self.qw.ctypes.data,
                                   self.qw.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.q_d.data.ptr, self.q.ctypes.data,
                                   self.q.nbytes, cp.cuda.runtime.memcpyHostToDevice)

    def solve(self):
        """
        Parameters
        ----------
        """

        self.time_step_size()
        dt_2 = self.dt / 2.0
        print('dt =', "{:.3e}".format(self.dt), end=' ')

        self.set_E()
        self.ME = self.Minv @ self.E

        # Solve u = A^-1 * b
        if not self.gpu:

            # intermediate velocity LHS
            A_vel = self.I + dt_2 * self.A + dt_2 * self.ME

            # intermediate x-velocity equation RHS
            self.b = self.u - dt_2 * self.A @ self.u + \
                self.dt * self.qu - A_vel @ self.u_dirichlet
            if self.sparse:
                K_sparse = sp.sparse.csr_matrix(
                    A_vel[self.points_to_solve_u, :][:, self.points_to_solve_u])
                self.u_star[self.points_to_solve_u, 0], exitCode = sp.sparse.linalg.gmres(
                    K_sparse, self.b[self.points_to_solve_u], x0=self.u_star[self.points_to_solve_u, 0])
            else:
                self.u_star[self.points_to_solve_u] = sp.linalg.solve(
                    A_vel[self.points_to_solve_u, :][:, self.points_to_solve_u], self.b[self.points_to_solve_u])

            # intermediate y-velocity equation RHS
            self.b = self.v - dt_2 * self.A @ self.v + \
                self.dt * self.qv - A_vel @ self.v_dirichlet
            if self.sparse:
                K_sparse = sp.sparse.csr_matrix(
                    A_vel[self.points_to_solve_v, :][:, self.points_to_solve_v])
                self.v_star[self.points_to_solve_v, 0], exitCode = sp.sparse.linalg.gmres(
                    K_sparse, self.b[self.points_to_solve_v], x0=self.v_star[self.points_to_solve_v, 0])
            else:
                self.v_star[self.points_to_solve_v] = sp.linalg.solve(
                    A_vel[self.points_to_solve_v, :][:, self.points_to_solve_v], self.b[self.points_to_solve_v])

            # intermediate z-velocity equation RHS
            self.b = self.w - dt_2 * self.A @ self.w + \
                self.dt * self.qw - A_vel @ self.w_dirichlet
            if self.sparse:
                K_sparse = sp.sparse.csr_matrix(
                    A_vel[self.points_to_solve_w, :][:, self.points_to_solve_w])
                self.w_star[self.points_to_solve_w, 0], exitCode = sp.sparse.linalg.gmres(
                    K_sparse, self.b[self.points_to_solve_w], x0=self.w_star[self.points_to_solve_w, 0])
            else:
                self.w_star[self.points_to_solve_w] = sp.linalg.solve(
                    A_vel[self.points_to_solve_w, :][:, self.points_to_solve_w], self.b[self.points_to_solve_w])

            # pressure equation RHS
            self.b = self.q - (1.0/self.dt) * (self.G1 @ self.u_star +
                                               self.G2 @ self.v_star + self.G3 @ self.w_star)

            # solve pressure equation
            if self.sparse:
                K_sparse = sp.sparse.csr_matrix(
                    self.K[self.points_to_solve, :][:, self.points_to_solve])
                self.p[self.points_to_solve, 0], exitCode = sp.sparse.linalg.gmres(
                    K_sparse, self.b[self.points_to_solve], x0=self.p[self.points_to_solve, 0])
            else:
                self.p[self.points_to_solve] = sp.linalg.solve(
                    self.K[self.points_to_solve, :][:, self.points_to_solve], self.b[self.points_to_solve])

            # update velocities
            self.u = self.u_star - self.dt * self.MG1 @ self.p
            self.v = self.v_star - self.dt * self.MG2 @ self.p
            self.w = self.w_star - self.dt * self.MG3 @ self.p

        else:

            # host to device data transfer
            cp.cuda.runtime.memcpy(self.ME_d.data.ptr, self.ME.ctypes.data,
                                   self.ME.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            # intermediate velocity LHS
            A_vel = self.I_d + dt_2 * self.A_d + dt_2 * self.ME_d

            # intermediate x-velocity equation RHS
            self.b_d = self.u_d - dt_2 * self.A_d @ self.u_d + \
                self.dt * self.qu_d - A_vel @ self.u_dirichlet_d
            if self.sparse:
                K_d_sparse = cps.sparse.csr_matrix(
                    A_vel[self.points_to_solve_u_d, :][:, self.points_to_solve_u_d])
                self.u_star_d[self.points_to_solve_u_d, 0], exitCode = cps.sparse.linalg.gmres(
                    K_d_sparse, self.b_d[self.points_to_solve_u_d], x0=self.u_star_d[self.points_to_solve_u_d, 0])
            else:
                self.u_star_d[self.points_to_solve_u_d] = cp.linalg.solve(
                    A_vel[self.points_to_solve_u_d, :][:, self.points_to_solve_u_d], self.b_d[self.points_to_solve_u_d])

            # intermediate y-velocity equation RHS
            self.b_d = self.v_d - dt_2 * self.A_d @ self.v_d + \
                self.dt * self.qv_d - A_vel @ self.v_dirichlet_d
            if self.sparse:
                K_d_sparse = cps.sparse.csr_matrix(
                    A_vel[self.points_to_solve_v_d, :][:, self.points_to_solve_v_d])
                self.v_star_d[self.points_to_solve_v_d, 0], exitCode = cps.sparse.linalg.gmres(
                    K_d_sparse, self.b_d[self.points_to_solve_v_d], x0=self.v_star_d[self.points_to_solve_v_d, 0])
            else:
                self.v_star_d[self.points_to_solve_v_d] = cp.linalg.solve(
                    A_vel[self.points_to_solve_v_d, :][:, self.points_to_solve_v_d], self.b_d[self.points_to_solve_v_d])

            # intermediate y-velocity equation RHS
            self.b_d = self.w_d - dt_2 * self.A_d @ self.w_d + \
                self.dt * self.qw_d - A_vel @ self.w_dirichlet_d
            if self.sparse:
                K_d_sparse = cps.sparse.csr_matrix(
                    A_vel[self.points_to_solve_w_d, :][:, self.points_to_solve_w_d])
                self.w_star_d[self.points_to_solve_w_d, 0], exitCode = cps.sparse.linalg.gmres(
                    K_d_sparse, self.b_d[self.points_to_solve_w_d], x0=self.w_star_d[self.points_to_solve_w_d, 0])
            else:
                self.w_star_d[self.points_to_solve_w_d] = cp.linalg.solve(
                    A_vel[self.points_to_solve_w_d, :][:, self.points_to_solve_w_d], self.b_d[self.points_to_solve_w_d])

            # pressure equation RHS
            self.b_d = self.q_d - (1.0/self.dt) * (self.G1_d @ self.u_star_d +
                                                   self.G2_d @ self.v_star_d + self.G3_d @ self.w_star_d)

            # solve pressure equation
            if self.sparse:
                K_d_sparse = cps.sparse.csr_matrix(
                    self.K_d[self.points_to_solve_d, :][:, self.points_to_solve_d])
                self.p_d[self.points_to_solve_d, 0], exitCode = cps.sparse.linalg.gmres(
                    K_d_sparse, self.b_d[self.points_to_solve_d], x0=self.p_d[self.points_to_solve_d, 0])
            else:
                self.p_d[self.points_to_solve_d] = cp.linalg.solve(
                    self.K_d[self.points_to_solve_d, :][:, self.points_to_solve_d], self.b_d[self.points_to_solve_d])

            # update velocities
            self.u_d = self.u_star_d - self.dt * self.MG1_d @ self.p_d
            self.v_d = self.v_star_d - self.dt * self.MG2_d @ self.p_d
            self.w_d = self.w_star_d - self.dt * self.MG3_d @ self.p_d

            # device to host data transfer
            cp.cuda.runtime.memcpy(self.u.ctypes.data, self.u_d.data.ptr,
                                   self.u_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)
            cp.cuda.runtime.memcpy(self.v.ctypes.data, self.v_d.data.ptr,
                                   self.v_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)
            cp.cuda.runtime.memcpy(self.w.ctypes.data, self.w_d.data.ptr,
                                   self.w_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)
            cp.cuda.runtime.memcpy(self.p.ctypes.data, self.p_d.data.ptr,
                                   self.p_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)

        # Set the known
        for key, value in self.mesh.bc_points_u["dirichlet"].items():
            self.u[self.mesh.pmap[key]] = value
            self.u_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_v["dirichlet"].items():
            self.v[self.mesh.pmap[key]] = value
            self.v_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_w["dirichlet"].items():
            self.w[self.mesh.pmap[key]] = value
            self.w_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points["dirichlet"].items():
            self.p[self.mesh.pmap[key]] = value

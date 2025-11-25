# FEM_NS
import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy as cps
from FEM_tri import GenericTriElement, GaussianQuadratureTri
# from FEM_mesh import Mesh_ns

# FEM Poisson 2D solver class
class FENS2D:
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
        G : numpy.ndarray
            x-Gradient matrix
        H : numpy.ndarray
            y-Gradient matrix
        E : numpy.ndarray
            Advection matrix
        A : numpy.ndarray
            A = Minv * K
        MG : numpy.ndarray
            MG = Minv * G
        MH : numpy.ndarray
            MH = Minv * H
        ME : numpy.ndarray
            ME = Minv * E
        s : numpy.ndarray
            Source vector for pressure
        su : numpy.ndarray
            source vector for x-velocity
        sv : numpy.ndarray
            source vector for y-velocity
        q : numpy.ndarray
            q = s - K * p_dirichlet
        qu : numpy.ndarray
            q = Minv * su
        qv : numpy.ndarray
            q = Minv * sv
        b : numpy.ndarray
            RHS
        u : numpy.ndarray
            x-velocity
        v : numpy.ndarray
            y-velocity
        p : numpy.ndarray
            pressure
        u_dirichlet : numpy.ndarray
            Dirichlet boundary x-velocity values
        v_dirichlet : numpy.ndarray
            Dirichlet boundary y-velocity values
        p_dirichlet : numpy.ndarray
            Dirichlet boundary pressure values
        u_star : numpy.ndarray
            intermediate x-velocity
        v_star : numpy.ndarray
            intermediate y-velocity
        points_to_solve : numpy.ndarray
            Index of points to solve for p
        points_to_solve_u : cupy.ndarray
            Index of points to solve for u
        points_to_solve_v : cupy.ndarray
            Index of points to solve for v   
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
        G_d : cupy.ndarray
            GPU G matrix
        H_d : cupy.ndarray
            GPU H matrix
        MG_d : cupy.ndarray
            GPU MG matrix
        MH_d : cupy.ndarray
            GPU MH matrix
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
        u_d : cupy.ndarray
            GPU x-velocity
        v_d : cupy.ndarray
            GPU y-velocity
        p_d : cupy.ndarray
            GPU pressure
        u_dirichlet_d : cupy.ndarray
            GPU u_dirichlet
        v_dirichlet_d : cupy.ndarray
            GPU v_dirichlet
        p_dirichlet_d : cupy.ndarray
            GPU p_dirichlet
        u_star_d : cupy.ndarray
            GPU u_star
        v_star_d : cupy.ndarray
            GPU v_star
        points_to_solve_d : cupy.ndarray
            GPU points_to_solve
        points_to_solve_u_d : cupy.ndarray
            GPU points_to_solve_u
        points_to_solve_v_d : cupy.ndarray
            GPU points_to_solve_v
        
    Methods
    -------
    time_step_size(self):
        Integration time step determination
    calc_local_update(self, p1, p2, p3):
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
    def __init__(self, _mesh, _f, _u=None, _v=None, _p=None, _gpu=False, _sparse=False, _dt=0.001):
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
        self.gte = GenericTriElement()
        self.gauss_quad = GaussianQuadratureTri()

        self.mesh = _mesh
        self.n_elements = self.mesh.tri.nsimplex
        self.n_points = self.mesh.npoints

        self.f = _f

        self.M = np.zeros((self.n_points, self.n_points))
        self.Minv = np.zeros_like(self.M)
        self.K = np.zeros_like(self.M)
        self.A = np.zeros_like(self.M)
        self.G = np.zeros_like(self.M)
        self.H = np.zeros_like(self.M)
        self.E = np.zeros_like(self.M)
        self.MG = np.zeros_like(self.M)
        self.MH = np.zeros_like(self.M)
        self.ME = np.zeros_like(self.M)
        self.I = np.eye(self.n_points)

        self.b = np.zeros((self.n_points, 1))
        self.su = np.zeros_like(self.b)
        self.sv = np.zeros_like(self.b)
        self.s = np.zeros_like(self.b)
        self.qu = np.zeros_like(self.b)
        self.qv = np.zeros_like(self.b)
        self.q = np.zeros_like(self.b)
        self.u_dirichlet = np.zeros_like(self.b)
        self.v_dirichlet = np.zeros_like(self.b)
        self.p_dirichlet = np.zeros_like(self.b)
        self.u_star = np.zeros_like(self.b)
        self.v_star = np.zeros_like(self.b)
        
        if _u is None:
            self.u = np.zeros_like(self.b)
        else:
            self.u = np.array(_u, copy=True)

        if _v is None:
            self.v = np.zeros_like(self.b)
        else:
            self.v = np.array(_v, copy=True)

        if _p is None:
            self.p = np.zeros_like(self.b)
        else:
            self.p = np.array(_p, copy=True)

        self.dt = _dt
        self.nu = 1

        self.points_to_solve   = np.array([], dtype=np.int32)
        self.points_to_solve_u = np.array([], dtype=np.int32)
        self.points_to_solve_v = np.array([], dtype=np.int32)

        self.gpu = _gpu
        self.sparse = _sparse
        if self.gpu:
            # Memory Pools for Efficient Allocation
            self.mp = cp.get_default_memory_pool()
            self.pp = cp.get_default_pinned_memory_pool()
            
            self.K_d = cp.zeros((self.n_points, self.n_points))
            self.I_d = cp.zeros_like(self.K_d)
            self.A_d = cp.zeros_like(self.K_d)
            
            self.G_d = cp.zeros_like(self.K_d)
            self.H_d = cp.zeros_like(self.K_d)
            self.MG_d = cp.zeros_like(self.K_d)
            self.MH_d = cp.zeros_like(self.K_d)
            self.ME_d = cp.zeros_like(self.K_d)
            
            self.b_d = cp.zeros((self.n_points, 1))
            self.qu_d = cp.zeros_like(self.b_d)
            self.qv_d = cp.zeros_like(self.b_d)
            self.q_d = cp.zeros_like(self.b_d)
            
            self.u_d = cp.zeros_like(self.b_d)
            self.v_d = cp.zeros_like(self.b_d)
            self.p_d = cp.zeros_like(self.b_d)

            self.u_star_d = cp.zeros_like(self.b_d)
            self.v_star_d = cp.zeros_like(self.b_d)
            
            self.u_dirichlet_d = np.zeros_like(self.b_d)
            self.v_dirichlet_d = np.zeros_like(self.b_d)
            self.p_dirichlet_d = np.zeros_like(self.b_d)
            
            self.points_to_solve_d   = np.array([], dtype=np.int32)
            self.points_to_solve_u_d = np.array([], dtype=np.int32)
            self.points_to_solve_v_d = np.array([], dtype=np.int32)

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
            p1, p2, p3 = (self.mesh.tri.points[el_ps[0]],
                            self.mesh.tri.points[el_ps[1]],
                            self.mesh.tri.points[el_ps[2]])
        
            d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
            d32 = (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2
            dmin2 = min(dmin2, d12, d13, d32)

        vel_max = max(np.abs(self.u).max(), np.abs(self.v).max()) + np.sqrt(1.4 * self.p.max())
        dt_conv = np.sqrt(dmin2) * CFL / vel_max
        dt_diff = dmin2
        self.dt = dt_conv # min(dt_diff, dt_conv)

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
        G_local = np.zeros_like(K_local)
        H_local = np.zeros_like(K_local)
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

        # local G and H matrix
        G_local[0, 0] = G_local[1, 0] = G_local[2, 0] = j22 / 6.0
        G_local[0, 1] = G_local[1, 1] = G_local[2, 1] = - j12 / 6.0
        G_local[0, 2] = G_local[1, 2] = G_local[2, 2] = (j12 - j22) / 6.0
        
        H_local[0, 0] = H_local[1, 0] = H_local[2, 0] = - j21 / 6.0
        H_local[0, 1] = H_local[1, 1] = H_local[2, 1] = j11 / 6.0
        H_local[0, 2] = H_local[1, 2] = H_local[2, 2] = (j21 - j11) / 6.0

        # b matrix
        b_local = j_det * self.gauss_quad.calculate(self.f, p1, p2, p3)

        return K_local, M_local, b_local, G_local, H_local

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
            K_local, M_local, _, G_local, H_local = self.calc_local_update(p1, p2, p3)

            # Assemble element's matrix solution into global matrix
            columns = np.array([self.mesh.pmap[el_ps] for _ in range(3)])
            rows = columns.T
            self.K[rows, columns] += K_local
            self.M[rows, columns] += M_local
            self.G[rows, columns] += G_local
            self.H[rows, columns] += H_local

    def set_E(self):
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
            xbar = np.zeros((1, 3))
            ybar = np.zeros((1, 3))
            ubar = np.zeros((3, 1))
            vbar = np.zeros((3, 1))
            
            xbar[0, :] = (1.0/24.0) * np.array([(p2[0]-p3[0]), -(p1[0]-p3[0]), -(p2[0]-p1[0])])
            ybar[0, :] = (1.0/24.0) * np.array([(p2[1]-p3[1]), -(p1[1]-p3[1]), -(p2[1]-p1[1])])

            ubar[:, 0] = np.array([ 2*self.u[self.mesh.pmap[el_ps[0]], 0] + self.u[self.mesh.pmap[el_ps[1]], 0]   + self.u[self.mesh.pmap[el_ps[2]], 0], 
                                    self.u[self.mesh.pmap[el_ps[0]], 0]   + 2*self.u[self.mesh.pmap[el_ps[1]], 0] + self.u[self.mesh.pmap[el_ps[2]], 0],
                                    self.u[self.mesh.pmap[el_ps[0]], 0]   + self.u[self.mesh.pmap[el_ps[1]], 0]   + 2*self.u[self.mesh.pmap[el_ps[2]], 0] ])
            
            vbar[:, 0] = np.array([ 2*self.v[self.mesh.pmap[el_ps[0]], 0] + self.v[self.mesh.pmap[el_ps[1]], 0]   + self.v[self.mesh.pmap[el_ps[2]], 0], 
                                    self.v[self.mesh.pmap[el_ps[0]], 0]   + 2*self.v[self.mesh.pmap[el_ps[1]], 0] + self.v[self.mesh.pmap[el_ps[2]], 0],
                                    self.v[self.mesh.pmap[el_ps[0]], 0]   + self.v[self.mesh.pmap[el_ps[1]], 0]   + 2*self.v[self.mesh.pmap[el_ps[2]], 0] ])

            E_local = ubar @ ybar - vbar @ xbar

            # Assemble element's matrix solution into global matrix
            columns = np.array([self.mesh.pmap[el_ps] for _ in range(3)])
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
            p1, p2, p3 = (self.mesh.tri.points[el_ps[0]],
                          self.mesh.tri.points[el_ps[1]],
                          self.mesh.tri.points[el_ps[2]])

            # Store local element's solution
            _, _, b_local, _, _ = self.calc_local_update(p1, p2, p3)

            # Assemble element's matrix solution into global matrix
            self.s[self.mesh.pmap[el_ps], 0] += b_local


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
        for key, value in self.mesh.bc_points_p["dirichlet"].items():
            self.p_dirichlet[self.mesh.pmap[key]] = value

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
                self.su[self.mesh.pmap[p_idx]] += 0.5 * distance * du_value  # du_boundary

        for ch_idx, du_values in self.mesh.bc_points_v["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.tri.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.sv[self.mesh.pmap[p_idx]] += 0.5 * distance * du_value  # du_boundary

        for ch_idx, du_values in self.mesh.bc_points_p["neumann_edge"].items():
            # convex_hull is a list with pair of point indices
            ch_points = self.mesh.tri.convex_hull[ch_idx]
            p1, p2 = self.mesh.tri.points[ch_points]
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            # Store the line integral in vector b
            for p_idx, du_value in zip(ch_points, du_values):
                self.s[self.mesh.pmap[p_idx]]  += 0.5 * distance * du_value  # du_boundary

    def initialze(self):
        """
        Parameters
        ----------
        p1, p2, p3: numpy.ndarray
            coordinates of a triangle
        """        
        # assign points to solve
        counter = 0
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points_p["dirichlet"] and self.mesh.pflg[p_idx]:
                self.points_to_solve = np.append(self.points_to_solve, self.mesh.pmap[p_idx])

        # assign points to solve
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points_u["dirichlet"] and self.mesh.pflg[p_idx]:
                self.points_to_solve_u = np.append(self.points_to_solve_u, self.mesh.pmap[p_idx])

        # assign points to solve
        for p_idx in range(self.mesh.tri.npoints):
            if p_idx not in self.mesh.bc_points_v["dirichlet"] and self.mesh.pflg[p_idx]:
                self.points_to_solve_v = np.append(self.points_to_solve_v, self.mesh.pmap[p_idx])

        if self.gpu:
            self.points_to_solve_d   = cp.asarray(self.points_to_solve)
            self.points_to_solve_u_d = cp.asarray(self.points_to_solve_u)
            self.points_to_solve_v_d = cp.asarray(self.points_to_solve_v)
        
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
        self.q  = self.s - self.K @ self.p_dirichlet

        self.A  = self.Minv @ self.K * self.nu
        self.MG = self.Minv @ self.G
        self.MH = self.Minv @ self.H

        # Set the known
        for key, value in self.mesh.bc_points_u["dirichlet"].items():
            self.u[self.mesh.pmap[key]] = value
            self.u_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_v["dirichlet"].items():
            self.v[self.mesh.pmap[key]] = value
            self.v_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_p["dirichlet"].items():
            self.p[self.mesh.pmap[key]] = value

        # host to device data transfer
        if self.gpu:

            cp.cuda.runtime.memcpy(self.u_d.data.ptr, self.u.ctypes.data, self.u.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.v_d.data.ptr, self.v.ctypes.data, self.v.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.p_d.data.ptr, self.p.ctypes.data, self.p.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            cp.cuda.runtime.memcpy(self.u_star_d.data.ptr, self.u_star.ctypes.data, self.u_star.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.v_star_d.data.ptr, self.v_star.ctypes.data, self.v_star.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            
            cp.cuda.runtime.memcpy(self.u_dirichlet_d.data.ptr, self.u_dirichlet.ctypes.data, self.u_dirichlet.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.v_dirichlet_d.data.ptr, self.v_dirichlet.ctypes.data, self.v_dirichlet.nbytes, cp.cuda.runtime.memcpyHostToDevice)
        
            cp.cuda.runtime.memcpy(self.I_d.data.ptr, self.I.ctypes.data, self.I.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.K_d.data.ptr, self.K.ctypes.data, self.K.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.A_d.data.ptr, self.A.ctypes.data, self.A.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            
            cp.cuda.runtime.memcpy(self.G_d.data.ptr, self.G.ctypes.data, self.G.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.H_d.data.ptr, self.H.ctypes.data, self.H.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.MG_d.data.ptr, self.MG.ctypes.data, self.MG.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.MH_d.data.ptr, self.MH.ctypes.data, self.MH.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            
            cp.cuda.runtime.memcpy(self.qu_d.data.ptr, self.qu.ctypes.data, self.qu.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.qv_d.data.ptr, self.qv.ctypes.data, self.qv.nbytes, cp.cuda.runtime.memcpyHostToDevice)
            cp.cuda.runtime.memcpy(self.q_d.data.ptr, self.q.ctypes.data, self.q.nbytes, cp.cuda.runtime.memcpyHostToDevice)

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
            self.b = self.u - dt_2 * self.A @ self.u + self.dt * self.qu - A_vel @ self.u_dirichlet
            if self.sparse:
                K_sparse = sp.sparse.csr_matrix(A_vel[self.points_to_solve_u, :][:, self.points_to_solve_u])
                self.u_star[self.points_to_solve_u, 0], exitCode = sp.sparse.linalg.gmres(K_sparse, self.b[self.points_to_solve_u], x0=self.u_star[self.points_to_solve_u, 0])
            else:
                self.u_star[self.points_to_solve_u] = sp.linalg.solve(A_vel[self.points_to_solve_u, :][:, self.points_to_solve_u], self.b[self.points_to_solve_u])

            # intermediate y-velocity equation RHS
            self.b = self.v - dt_2 * self.A @ self.v + self.dt * self.qv - A_vel @ self.v_dirichlet
            if self.sparse:
                K_sparse = sp.sparse.csr_matrix(A_vel[self.points_to_solve_v, :][:, self.points_to_solve_v])
                self.v_star[self.points_to_solve_v, 0], exitCode = sp.sparse.linalg.gmres(K_sparse, self.b[self.points_to_solve_v], x0=self.v_star[self.points_to_solve_v, 0])
            else:
                self.v_star[self.points_to_solve_v] = sp.linalg.solve(A_vel[self.points_to_solve_v, :][:, self.points_to_solve_v], self.b[self.points_to_solve_v])        
            
            # pressure equation RHS
            self.b = self.q - (1.0/self.dt) * (self.G @ self.u_star + self.H @ self.v_star)

            # solve pressure equation
            if self.sparse:
                K_sparse = sp.sparse.csr_matrix(self.K[self.points_to_solve, :][:, self.points_to_solve])
                self.p[self.points_to_solve, 0], exitCode = sp.sparse.linalg.gmres(K_sparse, self.b[self.points_to_solve], x0=self.p[self.points_to_solve, 0])
            else:
                self.p[self.points_to_solve] = sp.linalg.solve(self.K[self.points_to_solve, :][:, self.points_to_solve], self.b[self.points_to_solve])

            # update velocities
            self.u = self.u_star - self.dt * self.MG @ self.p
            self.v = self.v_star - self.dt * self.MH @ self.p
            
        else: 
            
            # host to device data transfer
            cp.cuda.runtime.memcpy(self.ME_d.data.ptr, self.ME.ctypes.data, self.ME.nbytes, cp.cuda.runtime.memcpyHostToDevice)

            # intermediate velocity LHS
            A_vel = self.I_d + dt_2 * self.A_d + dt_2 * self.ME_d

            # intermediate x-velocity equation RHS
            self.b_d = self.u_d - dt_2 * self.A_d @ self.u_d + self.dt * self.qu_d - A_vel @ self.u_dirichlet_d
            if self.sparse:
                K_d_sparse = cps.sparse.csr_matrix(A_vel[self.points_to_solve_u_d, :][:, self.points_to_solve_u_d])
                self.u_star_d[self.points_to_solve_u_d, 0], exitCode = cps.sparse.linalg.gmres(K_d_sparse, self.b_d[self.points_to_solve_u_d], x0=self.u_star_d[self.points_to_solve_u_d, 0])
            else: 
                self.u_star_d[self.points_to_solve_u_d] = cp.linalg.solve(A_vel[self.points_to_solve_u_d, :][:, self.points_to_solve_u_d], self.b_d[self.points_to_solve_u_d])

            # intermediate y-velocity equation RHS
            self.b_d = self.v_d - dt_2 * self.A_d @ self.v_d + self.dt * self.qv_d - A_vel @ self.v_dirichlet_d
            if self.sparse:
                K_d_sparse = cps.sparse.csr_matrix(A_vel[self.points_to_solve_v_d, :][:, self.points_to_solve_v_d])
                self.v_star_d[self.points_to_solve_v_d, 0], exitCode = cps.sparse.linalg.gmres(K_d_sparse, self.b_d[self.points_to_solve_v_d], x0=self.v_star_d[self.points_to_solve_v_d, 0])
            else: 
                self.v_star_d[self.points_to_solve_v_d] = cp.linalg.solve(A_vel[self.points_to_solve_v_d, :][:, self.points_to_solve_v_d], self.b_d[self.points_to_solve_v_d])

            # pressure equation RHS
            self.b_d = self.q_d - (1.0/self.dt) * (self.G_d @ self.u_star_d + self.H_d @ self.v_star_d)

            # solve pressure equation
            if self.sparse:
                K_d_sparse = cps.sparse.csr_matrix(self.K_d[self.points_to_solve_d, :][:, self.points_to_solve_d])
                self.p_d[self.points_to_solve_d, 0], exitCode = cps.sparse.linalg.gmres(K_d_sparse, self.b_d[self.points_to_solve_d], x0=self.p_d[self.points_to_solve_d, 0])
            else: 
                self.p_d[self.points_to_solve_d] = cp.linalg.solve(self.K_d[self.points_to_solve_d, :][:, self.points_to_solve_d], self.b_d[self.points_to_solve_d])

            # update velocities
            self.u_d = self.u_star_d - self.dt * self.MG_d @ self.p_d
            self.v_d = self.v_star_d - self.dt * self.MH_d @ self.p_d
            
            # device to host data transfer
            cp.cuda.runtime.memcpy(self.u.ctypes.data, self.u_d.data.ptr, self.u_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)
            cp.cuda.runtime.memcpy(self.v.ctypes.data, self.v_d.data.ptr, self.v_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)
            cp.cuda.runtime.memcpy(self.p.ctypes.data, self.p_d.data.ptr, self.p_d.nbytes, cp.cuda.runtime.memcpyDeviceToHost)

        # Set the known
        for key, value in self.mesh.bc_points_u["dirichlet"].items():
            self.u[self.mesh.pmap[key]] = value
            self.u_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_v["dirichlet"].items():
            self.v[self.mesh.pmap[key]] = value
            self.v_star[self.mesh.pmap[key]] = value
        for key, value in self.mesh.bc_points_p["dirichlet"].items():
            self.p[self.mesh.pmap[key]] = value
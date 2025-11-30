# FEM_tri
import numpy as np
import scipy as sp
import alphashape as ashp

# Triangle class
class GenericElement:
    """
    class for 2D basis for a triangular element

    ...

    Attributes
    ----------
        
    Methods
    -------
    N1(xi, eta, tau)
        N1 basis
        
    N2(xi, eta, tau)
        N2 basis
        
    N3(xi, eta, tau)
        N3 basis

    N4(xi, eta, tau)
        N4 basis

    get_xy(self, xi, eta, p1, p2, p3)
        Coordinate transformation local to global
    """     
    def __init__(self):
        """
        Parameters
        ----------

        """             
        # Shape functions derivatives in a local triangular element
        N1_dxi = 1
        N2_dxi = 0
        N3_dxi = 0
        N4_dxi = -1
        
        N1_deta = 0
        N2_deta = 1
        N3_deta = 0
        N4_deta = -1
        
        N1_dtau = 0
        N2_dtau = 0
        N3_dtau = 1
        N4_dtau = -1
        
        self.dN = np.array([
            [N1_dxi, N1_deta, N1_dtau],
            [N2_dxi, N2_deta, N2_dtau],
            [N3_dxi, N3_deta, N3_dtau],
            [N4_dxi, N4_deta, N4_dtau]])

    @staticmethod
    def N1(xi, eta, tau):
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        tau : float
            local coordinate
        """           
        return xi

    @staticmethod
    def N2(xi, eta, tau):
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        tau : float
            local coordinate
        """          
        return eta

    @staticmethod
    def N3(xi, eta, tau):
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        tau : float
            local coordinate
        """          
        return tau

    @staticmethod
    def N4(xi, eta, tau):
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        tau : float
            local coordinate
        """          
        return 1 - xi - eta - tau

    # Coordinate transformation local to global
    def get_xy(self, xi, eta, tau, p1, p2, p3, p4):
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        tau : float
            local coordinate
        p1, p2, p3, p4: numpy.ndarray
            coordinates of a tetrahedron
        """           
        return (p1[0] * self.N1(xi, eta, tau) + p2[0] * self.N2(xi, eta, tau) + p3[0] * self.N3(xi, eta, tau) + p4[0] * self.N4(xi, eta, tau),
                p1[1] * self.N1(xi, eta, tau) + p2[1] * self.N2(xi, eta, tau) + p3[1] * self.N3(xi, eta, tau) + p4[1] * self.N4(xi, eta, tau),
                p1[2] * self.N1(xi, eta, tau) + p2[2] * self.N2(xi, eta, tau) + p3[2] * self.N3(xi, eta, tau) + p4[2] * self.N4(xi, eta, tau))


class GaussianQuadrature:
    """
    class for Gaussian integration

    ...

    Attributes
    ----------
        
    Methods
    -------
    calculate(self, _f, p1, p2, p3)
        Calculate the numerical integration for each node
    """
    
    def __init__(self):
        """
        Parameters
        ----------
        """           
        # N=4
        # xa= [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105]; 
        # ya= [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685];
        # za= [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105];
        # wt= [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000]/6;
        self.wps = [(0.5854101966249685, 0.1381966011250105, 0.1381966011250105),  # weighted points 
                    (0.1381966011250105, 0.1381966011250105, 0.1381966011250105), 
                    (0.1381966011250105, 0.1381966011250105, 0.5854101966249685),
                    (0.1381966011250105, 0.5854101966249685, 0.1381966011250105)]
        self.ws = (1 / 24.0, 1 / 24.0, 1 / 24.0, 1 / 24.0) # weights
        self.tri_element = GenericElement()


    # Calculate the numerical integration for each node
    def calculate(self, _f, p1, p2, p3, p4):
        """
        Parameters
        ----------
        _f : function
            R.H.S
        p1, p2, p3, p4 : numpy.ndarray
            coordinates of a triangle
        """            
        # Get the global (x,y) coordinates at the weighted points
        xys = [self.tri_element.get_xy(wp[0], wp[1], wp[2], p1, p2, p3, p4) for wp in self.wps]

        return np.array([
            sum([w * _f(xy[0], xy[1], xy[2])[0] * self.tri_element.N1(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[1] * self.tri_element.N1(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[2] * self.tri_element.N1(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[0] * self.tri_element.N2(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[1] * self.tri_element.N2(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[2] * self.tri_element.N2(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[0] * self.tri_element.N3(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[1] * self.tri_element.N3(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[2] * self.tri_element.N3(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[0] * self.tri_element.N4(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[1] * self.tri_element.N4(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)]),
            sum([w * _f(xy[0], xy[1], xy[2])[2] * self.tri_element.N4(
                wp[0], wp[1], wp[2]) for w, wp, xy in zip(self.ws, self.wps, xys)])
            ])

class Tetrahedron:
    """
    class for generating tetrahedrons from FreeCAD mesh 

    ...

    Attributes
    ----------
    faces : numpy.ndarray
            coordinates of vertices of faces/triangles
    points : numpy.ndarray
            coordinates of points
    nsimplex : int
        number of faces/triangles
    npoints : int
        number of simplices
    simplices : numpy.ndarray
        indices of vertices of faces/triangles
    boundary_points : numpy.ndarray
        indices of boundary points
    convex_hull : numpy.ndarray
        boundary-point-indices of lines on the mesh boundary
        
    Methods
    -------

    """
    def __init__(self, _faces, alpha=34, fcavity=None):
        self.faces = _faces
        self.points = np.unique(self.faces, axis=0)
        self.nsimplex = self.faces.shape[0]//4
        self.npoints = self.points.shape[0]

        self.simplices = []
        start = 0
        end = 4
        for fidx in range(self.nsimplex):
            p1, p2, p3, p4 = self.faces[start:end, :]
            pidx1 = self.points.tolist().index(p1.tolist())
            pidx2 = self.points.tolist().index(p2.tolist())
            pidx3 = self.points.tolist().index(p3.tolist())
            pidx4 = self.points.tolist().index(p4.tolist())
            self.simplices.append([pidx1, pidx2, pidx3, pidx4])
            start += 4
            end += 4
        self.simplices = np.array(self.simplices, dtype=np.int32)

        # cancave hull determines mesh boundary better than convex hull
        boundary_points_coord = np.unique(ashp.alphashape(self.points, alpha).vertices, axis=0)
        
        boundary_points_idx = []
        for p in boundary_points_coord:
            pidx = self.points.tolist().index(p.tolist())
            boundary_points_idx.append(pidx)
        self.boundary_points = np.unique(boundary_points_idx)

        if fcavity is not None:
            for pidx, p in enumerate(self.points):
                if fcavity(p):
                    self.boundary_points = np.append(self.boundary_points, pidx)
            self.boundary_points = np.unique(self.boundary_points)

        X = self.points[self.boundary_points]
        dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) **2, axis=-1)
        nearest = np.argsort(dist_sq, axis=1)

        self.convex_hull = []
        for pidx, nidx, nnidx in self.boundary_points[nearest[:, :3]]:
            
            if not [nidx, pidx] in self.convex_hull:
                self.convex_hull.append([pidx, nidx])
            if not [nnidx, pidx] in self.convex_hull:
                self.convex_hull.append([pidx, nnidx])

        self.convex_hull = np.unique(self.convex_hull, axis=0)
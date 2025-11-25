# FEM_tri
import numpy as np
import scipy as sp
import shapely as shp

# Triangle class
class GenericTriElement:
    """
    class for 2D basis for a triangular element

    ...

    Attributes
    ----------
        
    Methods
    -------
    N1(xi, eta)
        N1 basis
        
    N2(xi, eta)
        N2 basis
        
    N3(xi, eta)
        N3 basis

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
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        """           
        return xi

    @staticmethod
    def N2(xi, eta):
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        """          
        return eta

    @staticmethod
    def N3(xi, eta):
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        """          
        return 1 - xi - eta

    # Coordinate transformation local to global
    def get_xy(self, xi, eta, p1, p2, p3):
        """
        Parameters
        ----------
        xi : float
            local coordinate
        eta : float
            local coordinate
        p1, p2, p3: numpy.ndarray
            coordinates of a triangle
        """           
        return (p1[0] * self.N1(xi, eta) + p2[0] * self.N2(xi, eta) + p3[0] * self.N3(xi, eta),
                p1[1] * self.N1(xi, eta) + p2[1] * self.N2(xi, eta) + p3[1] * self.N3(xi, eta))


class GaussianQuadratureTri:
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
        # nip = 3 # number of integration points
        self.wps = [(0.5, 0.5), (0.5, 0), (0, 0.5)]  # weighted points
        self.ws = (1 / 6, 1 / 6, 1 / 6)              # weights
        self.tri_element = GenericTriElement()

    # Calculate the numerical integration for each node
    def calculate(self, _f, p1, p2, p3):
        """
        Parameters
        ----------
        _f : function
            R.H.S
        p1, p2, p3: numpy.ndarray
            coordinates of a triangle
        """            
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

class Tri:
    """
    class for generating triangulation from FreeCAD mesh 

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
    def __init__(self, _faces, ratio=0.05, fcavity=None):
        self.faces = _faces
        self.points = np.unique(self.faces, axis=0)
        self.nsimplex = self.faces.shape[0]//3
        self.npoints = self.points.shape[0]

        self.simplices = []
        start = 0
        end = 3
        for fidx in range(self.nsimplex):
            p1, p2, p3 = self.faces[start:end, :]
            pidx1 = self.points.tolist().index(p1.tolist())
            pidx2 = self.points.tolist().index(p2.tolist())
            pidx3 = self.points.tolist().index(p3.tolist())
            self.simplices.append([pidx1, pidx2, pidx3])
            start += 3
            end += 3
        self.simplices = np.array(self.simplices, dtype=np.int32)

        # cancave hull determines mesh boundary better than convex hull
        multi_point = shp.MultiPoint(self.points)
        poly = shp.concave_hull(multi_point, ratio)
        xx, yy = poly.exterior.coords.xy
        boundary_points_coord = np.array(list(zip(xx, yy)))
        
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
        dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis=-1)
        nearest = np.argsort(dist_sq, axis=1)

        self.convex_hull = []
        for pidx, nidx, nnidx in self.boundary_points[nearest[:, :3]]:
            
            if not [nidx, pidx] in self.convex_hull:
                self.convex_hull.append([pidx, nidx])
            if not [nnidx, pidx] in self.convex_hull:
                self.convex_hull.append([pidx, nnidx])

        self.convex_hull = np.unique(self.convex_hull, axis=0)
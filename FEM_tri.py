# FEM_tri
import numpy as np
import scipy as sp

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
    def __init__(self, _faces, _boundary_simplices=None):
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

        self.convex_hull = sp.spatial.ConvexHull(self.points).simplices

        additional_convex_hull = []
        if _boundary_simplices is not None:
            start = 0
            end = 2
            for fidx in range(_boundary_simplices.shape[0]//2):
                p1, p2 = _boundary_simplices[start:end, :]
                pidx1 = self.points.tolist().index(p1.tolist())
                pidx2 = self.points.tolist().index(p2.tolist())
                additional_convex_hull.append([pidx1, pidx2])
                start += 2
                end += 2
        additional_convex_hull = np.array(additional_convex_hull, dtype=np.int32)

        self.boundary_points = []
        for pidx1, pidx2 in self.convex_hull:
            p1 = self.points[pidx1]
            p2 = self.points[pidx2]
            d12 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            for pidx3, p3 in enumerate(self.points):
                d13 = np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
                d23 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
                if np.abs(d12 - d13 - d23) < 1e-5:
                    self.boundary_points.append(pidx3)
        
        for pidx1, pidx2 in additional_convex_hull:
            p1 = self.points[pidx1]
            p2 = self.points[pidx2]
            d12 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            for pidx3, p3 in enumerate(self.points):
                d13 = np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
                d23 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
                if np.abs(d12 - d13 - d23) < 1e-5:
                    self.boundary_points.append(pidx3)

        self.boundary_points = np.unique(self.boundary_points)
        # print(self.boundary_points)

        X = self.points[self.boundary_points]
        dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis=-1)
        nearest = np.argsort(dist_sq, axis=1)
        # print(nearest[:, :3])

        self.convex_hull = []
        for pidx, nidx, nnidx in self.boundary_points[nearest[:, :3]]:
            # print(pidx, self.points[pidx], nidx, self.points[nidx], nnidx, nidx, self.points[nnidx])
            if not [nidx, pidx] in self.convex_hull:
                self.convex_hull.append([pidx, nidx])
            if not [nnidx, pidx] in self.convex_hull:
                self.convex_hull.append([pidx, nnidx])

        self.convex_hull = np.unique(self.convex_hull, axis=0)
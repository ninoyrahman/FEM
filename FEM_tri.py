# FEM_tri
import numpy as np

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
# Mesh class
import numpy as np
import scipy as sp
import shapely as shp
from FEM_tetrahedron_3d import Tetrahedron

class Mesh:
    """
    class for Mesh generator for square or circular plate

    ...

    Attributes
    ----------
        points : float
            grid points
        tri : float
            Delaunay triangles
        boundary_points : float
            boundary points index
        bc_points : float
            dict for dirichlet  boundary points and neumann boundary edges
    
    Methods
    -------
    
    """    
    def __init__(self, x_min, x_max, n_x, y_min, y_max, n_y, z_min, z_max, n_z, rout=None):
        """
        Parameters
        ----------
        x_min : float
            x-axes minimum
        x_max : float
            x-axes maximum
        n_x : int
            number of points along x-axes
        y_min : float
            y-axes minimum
        y_max : float
            y-axes maximum
        n_y : int
            number of points along y-axes
        rout : float, optional
            outer radius of circular plate, None for square plate
        """
        # Create a list with points coordinate (x,y)
        points = []
        nodes_x = np.linspace(x_min, x_max, n_x)
        nodes_y = np.linspace(y_min, y_max, n_y)
        nodes_z = np.linspace(z_min, z_max, n_z)
        
        for x in nodes_x:
            for y in nodes_y:
                for z in nodes_z: 
                    points.append([x, y, z])
                        
        points = np.array(points)
        self.points = points

        # Create Delaunay object
        self.tri = sp.spatial.Delaunay(points)

        # Identify the boundary points
        self.boundary_points = np.unique(self.tri.convex_hull.flatten())

        # Initialize the boundary conditions dictionary
        self.bc_points_u = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }
        self.bc_points_v = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }
        self.bc_points_w = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }

class Mesh_from_FreeCAD:
    """
    class for FreeCAD Mesh for SM equations

    ...

    Attributes
    ----------
        points : float
            grid points
        tri : float
            Delaunay triangles
        boundary_points : float
            boundary points index
        bc_points_u : float
            dict for dirichlet boundary points and neumann boundary edges for displacement
        bc_points_v : float
            dict for dirichlet boundary points and neumann boundary edges for displacement
    
    Methods
    -------
    
    """
    def __init__(self, _faces, alpha=34, fcavity=None):
        """
        Parameters
        ----------
        _faces : numpy.ndarray
            Simplices coordinates
        ratio : float, Optional
            ratio for concave hull, smaller value for higher refinement
        """

        # Create Triangulation
        self.tri = Tetrahedron(_faces, alpha, fcavity)
        self.points = self.tri.points

        # Identify the boundary points
        self.boundary_points = np.unique(self.tri.boundary_points)

        # Initialize the boundary conditions dictionary
        self.bc_points_u = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }
        self.bc_points_v = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }
        self.bc_points_w = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }
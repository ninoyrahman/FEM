# Mesh class
import numpy as np
import scipy as sp
import shapely as shp

class Mesh:
    def __init__(self, x_min, x_max, n_x, y_min, y_max, n_y, rout=None):
        # Create a list with points coordinate (x,y)
        points = []
        nodes_x = np.linspace(x_min, x_max, n_x)
        nodes_y = np.linspace(y_min, y_max, n_y)
        
        for x in nodes_x:
            for y in nodes_y:
                if rout is None: # square plate
                    points.append([x, y])
                else: # circular plate
                    if x**2 + y**2 <= rout**2:
                        points.append([x, y])
                        
        points = np.array(points)
        self.points = points

        # Create Delaunay object
        self.tri = sp.spatial.Delaunay(points)

        # Identify the boundary points
        self.boundary_points = np.unique(self.tri.convex_hull.flatten())

        # Initialize the boundary conditions dictionary
        self.bc_points = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }

class Mesh_from_outline:
    def __init__(self, x_min, x_max, n_x, y_min, y_max, n_y, outline):
        # Create a list with points coordinate (x,y)
        points = []
        nodes_x = np.linspace(x_min, x_max, n_x)
        nodes_y = np.linspace(y_min, y_max, n_y)

        polygon = shp.geometry.polygon.Polygon(outline)
        
        for x in nodes_x:
            for y in nodes_y:
                point = shp.geometry.Point(x, y)
                if polygon.contains(point):
                    points.append([x, y])
                        
        points = np.array(points)
        self.points = points

        # Create Delaunay object
        self.tri = sp.spatial.Delaunay(points)

        # Identify the boundary points
        self.boundary_points = np.unique(self.tri.convex_hull.flatten())

        # Initialize the boundary conditions dictionary
        self.bc_points = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }

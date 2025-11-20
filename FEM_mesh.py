# Mesh class
import numpy as np
import scipy as sp
import shapely as shp

class Mesh:
    """
    class for Mesh generator 

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
    def __init__(self, x_min, x_max, n_x, y_min, y_max, n_y, rout=None):
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
    """
    class for Mesh generator with outer boundary outline points

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
    def __init__(self, x_min, x_max, n_x, y_min, y_max, n_y, outline):
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
        outline : float
            outer boundary outline points
        """
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

# Mesh class
class Mesh_cavity:
    """
    class for Mesh generator with outer boundary outline points

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
        pflg : bool
            points-outside-cavity flag
        sflg : bool
            triangles-outside-cavity flag
        bflg : bool
            boundary-points flag
        cbflg : bool
            cavity-boundary-points flag    
        
    Methods
    -------
    
    """          
    def __init__(self, x_min, x_max, n_x, y_min, y_max, n_y, r=0.1, rout=None):
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
        r : float, optional
            inner radius of circular cavity, default value 0.1
        rout : float, optional
            outer radius of circular plate, None for square plate
        """          
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

        r2 = r**2
        
        #flags for cavity
        # True is 1 and False is 0
        self.pflg = np.ones(self.tri.npoints, dtype=bool) # points-outside-cavity flag
        self.sflg = np.ones(self.tri.nsimplex, dtype=bool) # triangles-outside-cavity flag
        self.bflg = np.zeros(self.tri.npoints, dtype=bool) # boundary-points flag
        self.cbflg = np.zeros(self.tri.npoints, dtype=bool) # cavity-boundary-points flag

        self.ncavity = 0
        for idx, p in enumerate(self.tri.points):
            if (p**2).sum() < r2:
                self.pflg[idx] = False
                self.ncavity += 1

        self.npoints = self.tri.npoints - self.ncavity

        print('# cavity points=', self.ncavity)
        print('# non-cavity points=', self.npoints)
        print('# boundary points excluding cavity=', self.boundary_points.size)
        
        for idx, p_idx in enumerate(self.tri.simplices):
            if not self.pflg[p_idx[0]] or not self.pflg[p_idx[1]] or not self.pflg[p_idx[2]]:
                self.sflg[idx] = False
                p1, p2, p3 = (self.tri.points[p_idx[0]], self.tri.points[p_idx[1]], self.tri.points[p_idx[2]])
                if (p1**2).sum() >= r2:
                    self.cbflg[p_idx[0]] = True
                    self.boundary_points = np.append(self.boundary_points, p_idx[0]*np.ones(1, dtype=int))
                if (p2**2).sum() >= r2:
                    self.cbflg[p_idx[1]] = True
                    self.boundary_points = np.append(self.boundary_points, p_idx[1]*np.ones(1, dtype=int))
                if (p3**2).sum() >= r2:
                    self.cbflg[p_idx[2]] = True
                    self.boundary_points = np.append(self.boundary_points, p_idx[2]*np.ones(1, dtype=int))
        
        self.boundary_points = np.unique(self.boundary_points)
        print('# boundary points including cavity=', self.boundary_points.size)

        for p_idx in self.boundary_points:
            self.bflg[p_idx] = True

        # map point index to vectors and matrix index
        self.pmap = -np.ones(self.tri.npoints, dtype=int)

        idx = 0
        for i in range(self.tri.npoints):
            if self.pflg[i]:
                self.pmap[i] = idx
                idx += 1
                
        # map vectors and matrix index to point index 
        # self.emap = np.array([np.argwhere(self.pmap == i)[0, 0] for i in range(self.npoints)], dtype=int)
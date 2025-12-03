# Mesh class
import numpy as np
import scipy as sp
import shapely as shp
import itertools
from FEM_tetrahedron_3d import Tetrahedron

class Mesh_from_FreeCAD:
    """
    class for FreeCAD Mesh for SM/Heat equations

    ...

    Attributes
    ----------
        npoints : int
            number of grid points
        tri : Tetrahedron object
            Tetrahedron
        boundary_points : numpy.ndarray
            boundary points index
        convex_hull : numpy.ndarray
            boundary-edge-points indices
        bc_points : dict
            dict for dirichlet boundary points and neumann boundary edges
        bc_points_u : dict
            dict for dirichlet boundary points and neumann boundary edges for displacement
        bc_points_v : dict
            dict for dirichlet boundary points and neumann boundary edges for displacement
        bc_points_w : dict
            dict for dirichlet boundary points and neumann boundary edges for displacement
        pflg : numpy.ndarray
            points-outside-cavity flag
        sflg : numpy.ndarray
            triangles-outside-cavity flag
        bflg : numpy.ndarray
            boundary-points flag
        cbflg : numpy.ndarray
            cavity-boundary-points flag
    
    Methods
    -------
    
    """
    def __init__(self, _faces, alpha=34, outline=None):
        """
        Parameters
        ----------
        _faces : numpy.ndarray
            Simplices coordinates
        alpha : float, optional
            alpha parameter for alphashape
        outline : numpy.ndarray
            2D (x,y) outline of cavity
        """

        # Create Triangulation
        self.tri = Tetrahedron(_faces, alpha)
        self.npoints = self.tri.npoints

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
        self.bc_points = {
            "dirichlet": dict(),
            "neumann_edge": dict()
        }

        # flags for cavity
        # True is 1 and False is 0
        self.pflg = np.ones(self.tri.npoints, dtype=bool) # points-outside-cavity flag
        self.sflg = np.ones(self.tri.nsimplex, dtype=bool) # triangles-outside-cavity flag
        self.bflg = np.zeros(self.tri.npoints, dtype=bool) # boundary-points flag
        self.cbflg = np.zeros(self.tri.npoints, dtype=bool) # cavity-boundary-points flag

        if outline is not None:
            polygon = shp.geometry.polygon.Polygon(outline)

            self.pflg = np.array([False if polygon.contains(shp.geometry.Point(p[:2])) else True for p in self.tri.points])
            self.ncavity = self.npoints - self.pflg.sum()
            self.npoints = self.tri.npoints - self.ncavity

            print('# cavity points=', self.ncavity)
            print('# non-cavity points=', self.npoints)
            print('# boundary points excluding cavity=', self.boundary_points.size)

            self.sflg = np.array([True if np.all(self.pflg[p_idx]) else False for p_idx in self.tri.simplices], dtype=bool)

            geo_indices = self.tri.simplices[~self.sflg].reshape(-1) # point-indices of fully or partly inside simplices inside geometry
            geo_points = self.tri.points[geo_indices] # points of simplices fully/partly inside geometry
            geo_point_objects = shp.points(geo_points[:, 0], geo_points[:, 1]) # object for points of simplices fully/partly inside geometry
            contains_results = polygon.contains(geo_point_objects) # mask for points inside geometry
            outside_points_indices = geo_indices[~contains_results] # indicies of points outside geometry
            
            self.cbflg[outside_points_indices] = True
            self.boundary_points = np.unique(np.append(self.boundary_points, outside_points_indices))
            print('# boundary points including cavity=', self.boundary_points.size)

        self.bflg[self.boundary_points] = True

        # map point index to vectors and matrix index
        counter = itertools.count()
        self.pmap = np.array([next(counter) if flag else -1 for flag in self.pflg], dtype=int)

        # Boundary edges
        X = self.tri.points[self.boundary_points]
        dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) **2, axis=-1)
        nearest = np.argsort(dist_sq, axis=1)

        self.convex_hull = []
        for pidx, nidx, nnidx in self.boundary_points[nearest[:, :3]]:
            
            if not [nidx, pidx] in self.convex_hull:
                self.convex_hull.append([pidx, nidx])
            if not [nnidx, pidx] in self.convex_hull:
                self.convex_hull.append([pidx, nnidx])

        self.convex_hull = np.unique(self.convex_hull, axis=0)
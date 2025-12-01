# Mesh class
import numpy as np
import scipy as sp
import shapely as shp
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
            self.ncavity = 0
            for idx, p in enumerate(self.tri.points):
                point = shp.geometry.Point(p[0], p[1])
                if polygon.contains(point): #if (p**2).sum() < r2: 
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
                    if not polygon.contains(shp.geometry.Point(p1[0], p1[1])):
                        self.cbflg[p_idx[0]] = True
                        self.boundary_points = np.append(self.boundary_points, p_idx[0]*np.ones(1, dtype=int))
                    if not polygon.contains(shp.geometry.Point(p2[0], p2[1])):
                        self.cbflg[p_idx[1]] = True
                        self.boundary_points = np.append(self.boundary_points, p_idx[1]*np.ones(1, dtype=int))
                    if not polygon.contains(shp.geometry.Point(p3[0], p3[1])):
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
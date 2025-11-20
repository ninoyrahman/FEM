# plot3d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

def plot3d(ax, data, varmin, varmax, x, y, label=r'u', cmap=cm.coolwarm, azim=130):
    var = np.array(data, copy=True)
    lrange = np.linspace(varmin, varmax, 11)
    var[var < varmin] = varmin
    var[var > varmax] = varmax

    # print(lrange)
    
    ax.view_init(elev=20, azim=azim, roll=0)
    
    surf = ax.plot_trisurf(x, y, var,
                           linewidth=0.2, antialiased=True, cmap=cmap, label="FE")
    
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    # ax.set_aspect('equal')
    
    cbar = plt.colorbar(surf, ax=ax, orientation="horizontal", pad=0.02, ticks=lrange)
    cbar.ax.set_xlabel(label)
    cbar.ax.xaxis.set_label_position('bottom')
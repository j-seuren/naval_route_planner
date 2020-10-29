import os
import pickle
import main
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np

from case_studies.demos import create_currents
from matplotlib import font_manager as fm
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from pathlib import Path

fontPropFP = "C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/tex-gyre-pagella.regular.otf"
fontProp = fm.FontProperties(fname=fontPropFP, size=9)

loadDir = Path('D:/output/')
os.chdir(loadDir)
fileName = 'demoRaw'


def plot_demo(save=False):
    currentDict = create_currents(1, returnDict=True)
    u, v, lons, lats = currentDict['u'], currentDict['v'], currentDict['lons'], currentDict['lats']

    with open(fileName, 'rb') as fh:
        (proc, raw) = pickle.load(fh)

    front = raw['fronts'][0][0]

    fig, ax = plt.subplots()
    m = navigation_area(ax, u, v, lons, lats)

    cmap = colorbar(m)

    for j, ind in enumerate(front.items):
        plot_ind(ind, m, label=None, cmap=cmap, alpha=0.7)

    ax.legend(loc='upper right', prop=fontProp)
    plt.grid()
    plt.xticks(fontproperties=fontProp)
    plt.yticks(fontproperties=fontProp)

    if save:
        fig.savefig('demoRoutes', dpi=300)
        fig.savefig('demoRoutes.pdf', bbox_inches='tight', pad_inches=.02)


def navigation_area(ax, uin, vin, lons, lats):
    extent = (-15, -5, 15, 5)
    left, bottom, right, top = extent
    m = Basemap(projection='merc', resolution='l', ax=ax,
                llcrnrlat=bottom, urcrnrlat=top, llcrnrlon=left, urcrnrlon=right)
    m.drawmapboundary()
    m.fillcontinents(zorder=2)
    m.drawcoastlines()
    m.drawparallels(np.arange(-90., 90., 10.), labels=[1, 0, 0, 0], fontsize=8)
    m.drawmeridians(np.arange(-180., 180., 10.), labels=[0, 0, 0, 1], fontsize=8)

    # Currents
    dLon = extent[2] - extent[0]
    dLat = extent[3] - extent[1]
    vLon = int(dLon * 4)
    vLat = int(dLat * 4)
    uRot, vRot, x, y = m.transform_vector(uin, vin, lons, lats, vLon, vLat, returnxy=True)
    Q = m.quiver(x, y, uRot, vRot, np.hypot(uRot, vRot),
                 pivot='mid', width=0.002, headlength=4, cmap='Blues', scale=90, ax=ax)
    ax.quiverkey(Q, 0.4, 1.1, 2, r'$2$ knots', labelpos='E')

    return m


def colorbar(m):
    cmap = cm.get_cmap('jet', 12)
    # cmapList = [cmap(i) for i in range(cmap.N)][1:-1]
    # cmap = cl.LinearSegmentedColormap.from_list('Custom cmap', cmapList, cmap.N - 2)

    vMin, dV = 8.8, 15.2 - 8.8

    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = m.colorbar(sm, norm=plt.Normalize(vmin=vMin, vmax=vMin + dV),
                    size=0.2, pad=0.05, location='right')
    nTicks = 6
    cb.ax.set_yticklabels(['%.1f' % round(vMin + i * dV / (nTicks - 1), 1) for i in range(nTicks)],
                          fontproperties=fontProp)
    cb.set_label('Nominal speed [kn]', rotation=270, labelpad=15, fontproperties=fontProp)

    return cmap


def plot_ind(ind, m, label, cmap, alpha):
    vMin, dV = 8.8, 15.2 - 8.8

    waypoints, speeds = zip(*ind)
    for i, leg in enumerate(zip(waypoints[:-1], waypoints[1:])):
        color = cmap((speeds[i] - vMin) / dV) if cmap != 'k' else cmap
        label = None if i > 0 else label
        m.drawgreatcircle(leg[0][0], leg[0][1], leg[1][0], leg[1][1], label=label, linewidth=1,
                          alpha=alpha, color=color, zorder=3)


plot_demo()
plt.show()

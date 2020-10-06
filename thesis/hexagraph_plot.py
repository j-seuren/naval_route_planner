import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

from data_config.hexagraph import Hexagraph
from data_config.navigable_area import NavigableAreaGenerator
from pathlib import Path

DIR = Path('D:/')

os.chdir('..')
graph_d = 4
graph_vd = 6
_resolution = 'i'
splits = 3
_par = {'res': _resolution,
        'splits': splits,
        'graphVarDens': graph_vd,
        'graphDens': graph_d,
        'avoidAntarctic': True,
        'avoidArctic': True}

# Load and pre-process shoreline and ECA polygons
navAreaGenerator = NavigableAreaGenerator(_par, DIR=DIR)
_treeDict = navAreaGenerator.get_shoreline_rtree()
_ecaTreeDict = navAreaGenerator.get_eca_rtree()
_bathTree = navAreaGenerator.get_bathymetry_rtree()

# Initialize "Hexagraph"
hexagraph = Hexagraph(_treeDict,
                 _ecaTreeDict,
                 _bathTree,
                 _par,
                 DIR)

# # Get distribution of graph edge lengths
# fig1, ax1 = plt.subplots()
#
# distances = [miles[2] for miles in graph.edges.data('miles')]
# # _, bins = np.histogram(distances, bins=100)
# ax1.boxplot(distances)

# ax1.hist(distances, bins=bins, histtype='bar')
# ax1.set_xlabel('edge arc length [nmi]')
# plt.show()

# Plot graphs
# fig, ax = hexagraph.plot_graph(draw='edges', showLongEdges=False)
fig, ax = hexagraph.plot_sphere_edges(elevationAngle=27, azimuthAngle=52)
# fig, ax = hexagraph.plot_sphere(elevationAngle=27, azimuthAngle=52)

# # Strait of Gibraltar setting
# minX, maxX, minY, maxY = -10.375, 2.575, 34.525, 41
# plt.xlim(minX, maxX)
# plt.ylim(minY, maxY)
# rect = patches.Rectangle((minX, minY), maxX - minX, maxY - minY, linewidth=1, edgecolor='k', facecolor='none')
# redRect = patches.Rectangle((-7.7, 35.1), 4.4, 1.7, linewidth=2, edgecolor='red', facecolor='none')
# ax.add_patch(rect)
# ax.add_patch(redRect)

# fig.set_size_inches(4, 3)
ax.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=1)
plt.margins(0, 0, 0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
ax.zaxis.set_major_locator(plt.NullLocator())

# fig.savefig(DIR / 'output/figures/INIT_3D_sphere_3.pdf', bbox_inches='tight', pad_inches=0)
print('saved graph')
plt.show()

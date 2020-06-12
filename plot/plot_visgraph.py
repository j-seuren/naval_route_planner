import pyvisgraph as vg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Load the visibility graph file
graph = vg.VisGraph()
graph.load('output\GSHHS_c_L1.graph')

count = 0
segs = []

for edge in graph.visgraph.edges:
    x = [edge.p1.lon, edge.p2.lon]
    y = [edge.p1.lat, edge.p2.lat]
    segs.append(((x[0], y[0]), (x[1], y[1])))

ln_coll = LineCollection(segs, linewidths=0.1)

ax = plt.gca()
ax.add_collection(ln_coll)
ax.set_xlim(-180, 180)
ax.set_ylim(-56, 84)
plt.draw()
plt.show()


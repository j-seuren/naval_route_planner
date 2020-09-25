import matplotlib.pyplot as plt
import os
import random
from shapely.geometry import Point
from data_config.navigable_area import NavigableAreaGenerator

os.chdir('..')

# Get polygons

_parameters = {'avoidAntarctic': True, 'avoidArctic': True, 'splits': 5, 'res': 'i'}
generator = NavigableAreaGenerator(_parameters)
geo = generator.get_shorelines(split=False)[3]
split_polys = generator.split_polygon(geo)

print(len(split_polys))

envArea = polyArea = 0
for poly in split_polys:
    envArea += poly.envelope.area
    polyArea += poly.area

print('% dead space split = ', 1 - polyArea / envArea)
print('% dead space original = ', 1 - geo.area / geo.envelope.area)


def generate_random(number, polygon):
    points, envelopePoints = [], []
    minx, miny, maxx, maxy = polygon.bounds
    for i in range(number):
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
        else:
            envelopePoints.append(pnt)
    return points, envelopePoints


nPoints = 1000
ptsInUSA, envPts = generate_random(nPoints, geo)
print('{} of {} points lie within USA ({} lie outside USA)'.format(len(ptsInUSA), nPoints, len(envPts)))


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for ax in [ax1, ax2]:
    ax.axis('equal')
    ax.axis('off')
    ax.fill(*geo.exterior.xy, alpha=0.2, facecolor='gray')
    ax.plot(*geo.exterior.xy, color='gray', linewidth=1)
    ax.margins(0, 0)
    # Plot points in USA
    xs = [point.x for point in ptsInUSA]
    ys = [point.y for point in ptsInUSA]
    ax.scatter(xs, ys, c='blue', s=2)

# Plot MBR USA
ax1.plot(*geo.envelope.exterior.xy, color='black', linewidth=1)

xs = [point.x for point in envPts]
ys = [point.y for point in envPts]
ax1.scatter(xs, ys, c='red', s=2)  # Points in MBR

# Points in split MBRs
ptsInSplitMBRs = []
for pt in envPts:
    for geo in split_polys:
        if geo.envelope.contains(pt):
            ptsInSplitMBRs.append(pt)
xs = [point.x for point in ptsInSplitMBRs]
ys = [point.y for point in ptsInSplitMBRs]
ax2.scatter(xs, ys, c='red', s=2)

fracPtsInSplitMBRs = (len(ptsInUSA) + len(ptsInSplitMBRs)) / nPoints
print('Fraction of points that lie within (outside) split MBRs: {} ({})'.format(fracPtsInSplitMBRs, 1 - fracPtsInSplitMBRs))
print('Points in split dead space: ', len(ptsInSplitMBRs))

# Points outside split MBRs
ptsOutsideSplitMBRs = [pt for pt in envPts if pt not in ptsInSplitMBRs]
xs, ys = [point.x for point in ptsOutsideSplitMBRs], [point.y for point in ptsOutsideSplitMBRs]
ax2.scatter(xs, ys, c='gray', s=2)

for splGeo in split_polys:
    ax2.plot(*splGeo.envelope.exterior.xy, color='black', linewidth=1)

for i, fig in enumerate([fig1, fig2]):
    fig.savefig("filename.pdf", bbox_inches='tight', pad_inches=0)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # fig.savefig('thesis/RTREE_splitPolysUSA{}.pdf'.format(i), bbox_inches='tight', pad_inches=0)

# plt.show()
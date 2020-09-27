import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
import os
import pandas as pd
import pickle
import random
import time

from data_config.navigable_area import NavigableAreaGenerator, populate_rtree
from matplotlib.collections import LineCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import LineString
# from shapely.prepared import prep


os.chdir('..')


def get_polygons(parameters):
    # Get polygons
    generator = NavigableAreaGenerator(parameters)
    geos = generator.get_shorelines(split=False)
    splitGeos = generator.split_polygons(geos)

    print('splitGeos/geos', len(splitGeos) / len(geos),
          'splitGeos - geos', len(splitGeos) - len(geos),
          'length geos', len(geos),
          'length splitGeos', len(splitGeos))

    return geos, splitGeos


def poly_stats(geos, splitGeos, save=False):
    # Create histograms for (split)polygon edges and area
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey='row')
    geosEdges = [len(list(poly.exterior.coords)) for poly in geos]
    geosAreas = [poly.area * 1000 for poly in geos]

    print('Number of geos with area >25: ', len([geo for geo in geos if geo.area >= 25]))

    splitEdges = np.array([len(list(poly.exterior.coords)) for poly in splitGeos])
    splitAreas = np.array([poly.area for poly in splitGeos])

    print('Creating bins')
    _, ebins = np.histogram(geosEdges, bins=6)
    _, abins = np.histogram(geosAreas, bins=6)
    print('done')

    edgeBins = np.logspace(np.log10(ebins[0]), np.log10(ebins[-1]), len(ebins))
    areaBins = np.logspace(np.log10(abins[0]), np.log10(abins[-1]), len(abins))

    edges = [geosEdges, splitEdges]
    areas = [geosAreas, splitAreas]

    colors = ['red', 'blue']
    labels = ['before', 'after']
    ax0.hist(edges, bins=edgeBins, histtype='bar', color=colors, label=labels, log=True)
    ax0.set_xlabel('# edges')
    ax1.hist(areas, bins=areaBins, histtype='bar', color=colors, label=labels)
    ax1.set_xlabel('polygon size')

    ax1.legend()
    ax0.set_ylabel('# Polygons')

    for ax in (ax0, ax1):
        ax.set_xscale("log")
        ax.set_yscale("log")

    if save:
        tikzplotlib.save("D:/output/figures/histogram.tex")


# Generate n uniformly random distributed and oriented line segments
def generate_random(n):
    margin = 1
    lines = []
    minx, miny, maxx, maxy = -180+margin, -90+margin, 180-margin, 90-margin
    for n in range(n):
        x1 = random.uniform(minx, maxx)
        y1 = random.uniform(miny, maxy)
        R = random.random()
        if R < 0.25:
            x2 = x1 + random.uniform(margin/2, margin)
            y2 = y1 + random.uniform(margin/2, margin)
        elif R < 0.5:
            x2 = x1 - random.uniform(margin / 2, margin)
            y2 = y1 + random.uniform(margin / 2, margin)
        elif R < 0.75:
            x2 = x1 + random.uniform(margin / 2, margin)
            y2 = y1 - random.uniform(margin / 2, margin)
        else:
            x2 = x1 - random.uniform(margin / 2, margin)
            y2 = y1 - random.uniform(margin / 2, margin)

        line = LineString([(x1, y1), (x2, y2)])
        lines.append(line)
    return lines


def time_naive(lines, geos):
    # NAIVE APPROACH
    t0 = time.time()
    ptsInGeo = tests = nEdges = 0
    for line in lines:
        for geo in geos:
            tests += 1
            nEdges += len(list(geo.exterior.coords))
            if geo.intersects(line):
                ptsInGeo += 1
                break
    t1 = time.time()

    print('NAIVE: time-', t1 - t0,
          'pts in geo-', ptsInGeo,
          'calculations-', tests,
          'avg. edges-', nEdges / tests)


def time_rtree(lines, geos, printOutput=True):
    t0 = time.time()

    # Populate R-tree index with bounds of polygons
    treeSplitGeoDict = populate_rtree(geos)
    intersections = []
    ptsInGeo = tests = nEdges = nExtents = lenExtents = 0
    for line in lines:
        extent_intersections = treeSplitGeoDict['tree'].query(line)
        if extent_intersections:
            nExtents += 1
            lenExtents += len(extent_intersections)
            for geom in extent_intersections:
                tests += 1
                nEdges += len(geom.exterior.coords.xy[0])
                # geomIdx = treeSplitGeoDict['indexByID'][id(geom)]
                # prepGeom = treeSplitGeoDict['polys'][geomIdx]
                if geom.intersects(line):
                    ptsInGeo += 1
                    intersections.append(line)
                    break
    t1 = time.time()

    if printOutput:
        print('RTree:\n   time-', t1-t0,
              '\n   pts in geo-', ptsInGeo,
              '\n   calculations-', tests,
              '\n   avg. edges-', nEdges / tests,
              '\n   length geos', len(geos))

        print('avg M: ', lenExtents / nExtents)

    return intersections


def plot_intersections(lines, intersects, save=True, width=2):
    # Plot on Basemap
    fig, ax = plt.subplots()
    fig.set_size_inches(5.88, 3)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax)
    m.drawcoastlines(color='gray')
    # m.drawmapboundary()
    m.fillcontinents(alpha=0.2, color='gray')

    lon1s, lon2s, lat1s, lat2s = [], [], [], []

    for intersect in intersects:
        [lon1, lat1], [lon2, lat2] = intersect.xy
        lon1s.append(lon1)
        lon2s.append(lon2)
        lat1s.append(lat1)
        lat2s.append(lat2)

    dfInt = pd.DataFrame({"lon1": lon1s,
                          "lat1": lon2s,
                          "lon2": lat1s,
                          "lat2": lat2s})

    lon1sInt, lat1sInt = m(dfInt.lon1.values, dfInt.lat1.values)
    lon2sInt, lat2sInt = m(dfInt.lon2.values, dfInt.lat2.values)

    ptsInt = np.c_[lon1sInt, lat1sInt, lon2sInt, lat2sInt].reshape(len(lon1sInt), 2, 2)
    plt.gca().add_collection(LineCollection(ptsInt, color="red", linewidths=width))

    noIntersects = [line for line in lines if line not in intersects]

    lon1s, lon2s, lat1s, lat2s = [], [], [], []

    for restLine in noIntersects:
        [lon1, lat1], [lon2, lat2] = restLine.xy
        lon1s.append(lon1)
        lon2s.append(lon2)
        lat1s.append(lat1)
        lat2s.append(lat2)

    dfRest = pd.DataFrame({"lon1": lon1s,
                           "lat1": lon2s,
                           "lon2": lat1s,
                           "lat2": lat2s})

    lon1sRest, lat1sRest = m(dfRest.lon1.values, dfRest.lat1.values)
    lon2sRest, lat2sRest = m(dfRest.lon2.values, dfRest.lat2.values)

    ptsRest = np.c_[lon1sRest, lat1sRest, lon2sRest, lat2sRest].reshape(len(lon1sRest), 2, 2)
    plt.gca().add_collection(LineCollection(ptsRest, color="gray", linewidths=width))

    if save:
        fp = 'D:/output/figures/RTREE_linesegs_map.pdf'
        fig.savefig(fp,  bbox_inches='tight', pad_inches=0)
        print('saved to: ', fp)


def test_split_size(minSplit, maxSplit, nTests, nLines):
    nSplits = int(2 * maxSplit / minSplit) - 1  # steps of minsSplit / 2
    splitRange = np.linspace(minSplit, maxSplit, nSplits)
    print(splitRange)

    splitFP = 'thesis/splitPolygons_start{}_stop{}_n{}'.format(minSplit, maxSplit, nSplits)
    excelFP = 'thesis/averageTimes_start{}_stop{}_n{}.xlsx'.format(minSplit, maxSplit, nSplits)
    if os.path.exists(splitFP):
        with open(splitFP, 'rb') as file:
            splitGeosList = pickle.load(file)
    else:
        splitGeosList = []
        for splits in splitRange:
            _parameters = {'avoidAntarctic': False,
                           'avoidArctic': False,
                           'splits': splits,
                           'res': 'i'}

            _geos, _splitGeos = get_polygons(_parameters)
            splitGeosList.append(_splitGeos)

        with open(splitFP, 'wb') as file:
            pickle.dump(splitGeosList, file)

    print(len(splitGeosList), len(splitRange))

    totTimes = np.zeros(nSplits)
    for j in range(nTests):
        print('test', j+1)
        _lines = generate_random(nLines)
        # print('{} line segments generated'.format(nLines))
        times = []
        test0 = time.time()
        for i, split in enumerate(splitRange):
            print('\rsplit', split, end='')
            T0 = time.time()
            _intersections = time_rtree(_lines, splitGeosList[i], printOutput=False)
            times.append(time.time() - T0)

        times = np.array(times)
        totTimes += times
        print('\rtest time:', time.time() - test0)

        if j == 0:
            df = pd.DataFrame(totTimes)
            df.to_excel('thesis/firstAvg.xlsx', index=False)

    averageTimes = totTimes/nTests

    df = pd.DataFrame(averageTimes)
    df.to_excel(excelFP, index=False)

    print(totTimes/nTests)


MIN_SPLIT, MAX_SPLIT, N_TESTS = 0.5, 20, 100
N_LINES = 1000
SPLITS = 3

_parameters = {'avoidAntarctic': False,
               'avoidArctic': False,
               'splits': SPLITS,
               'res': 'i'}

LINES = generate_random(N_LINES)
GEOS, SPLIT_GEOS = get_polygons(_parameters)

time_naive(LINES, GEOS)
time_naive(LINES, SPLIT_GEOS)
time_rtree(LINES, GEOS)
time_rtree(LINES, SPLIT_GEOS)
# test_split_size(MIN_SPLIT, MAX_SPLIT, N_TESTS, N_LINES)
# plot_intersections(_lines, _intersections, save=True)
# plt.show()

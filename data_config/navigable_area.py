import matplotlib.pyplot as plt
import fiona
import os
import pickle

from pathlib import Path
from rtree.index import Index
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, shape, LineString, Point


class NavigableAreaGenerator:
    def __init__(self, parameters, DIR=Path('D:/')):
        self.resolution = parameters['res']
        self.splitThreshold = parameters['splits']
        self.avoidAntarctic = parameters['avoidAntarctic']
        self.avoidArctic = parameters['avoidArctic']
        self.gshhgDir = DIR / 'data/GSHHS_shp'
        self.shorelinesFP = self.gshhgDir / '{0}/GSHHS_{0}_L1.shp'.format(self.resolution)
        self.antarcticFP = self.gshhgDir / '{0}/GSHHS_{0}_L6.shp'.format(self.resolution)
        self.DIR = DIR

    def get_shoreline_rtree(self, getExterior=False):
        aC = 'excl' if self.avoidArctic else 'incl'
        aAc = 'excl' if self.avoidAntarctic else 'incl'
        fp = self.DIR / 'data/navigation_area/shorelines_{}_split{}_{}Antarc_{}Arc'.format(self.resolution, self.splitThreshold,
                                                                                   aAc, aC)
        shorelines = self.get_shorelines(fp, not getExterior)
        if getExterior:
            exteriors = []
            for polygon in shorelines:
                cutLines = cut(polygon.exterior, self.splitThreshold)
                exterior = [cutLines[0]]
                while len(cutLines) > 1:
                    cutLines = cut(cutLines[-1], 10)
                    exterior.append(cutLines[0])
                exteriors.extend(exterior)
            shorelines = exteriors

        if os.path.exists(fp / '.idx'):
            return {'rtree': Index(fp), 'geometries': shorelines}
        else:
            return populate_rtree(shorelines, fp)

    def get_bathymetry_rtree(self):
        fp = self.DIR / 'data/navigation_area/bath_split{}'.format(self.splitThreshold)
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                bathPolys = pickle.load(f)
        else:
            bathFP = self.DIR / 'data/bathymetry_200m/ne_10m_bathymetry_K_200.shp'
            polygons = [shape(polygon['geometry']) for polygon in iter(fiona.open(bathFP))]

            # Split and save polygons
            bathPolys = self.split_polygons(polygons)
            with open(fp, 'wb') as f:
                pickle.dump(bathPolys, f)
            print('Saved to: ', fp)

        if os.path.exists(fp / '.idx'):
            return {'rtree': Index(fp), 'geometries': bathPolys}
        else:
            return populate_rtree(bathPolys, fp)

    def get_shorelines(self, fp, split):
        if split and os.path.exists(fp):
            with open(fp, 'rb') as f:
                return pickle.load(f)
        shorelines = [shape(shoreline['geometry']) for shoreline in iter(fiona.open(self.shorelinesFP))]

        # If Antarctic circle is to be avoided; include as impassable area (latitude < -66),
        if self.avoidAntarctic:
            antarcticCircle = Polygon([(-181, -66), (181, -66), (181, -91), (-181, -91)])
            shorelines.append(antarcticCircle)
        else:  # Otherwise; include Antarctica in shoreline list
            antarctica = [shape(shoreline['geometry']) for shoreline in iter(fiona.open(self.antarcticFP))]
            shorelines.extend(antarctica)

        # If Arctic circle is to be avoided; include as impassable area (latitude > 66)
        if self.avoidArctic:
            arcticCircle = Polygon([(-181, 66), (181, 66), (181, 91), (-181, 91)])
            shorelines.append(arcticCircle)
        if split:
            splitShorelines = self.split_polygons(shorelines)
            # Save result
            with open(fp, 'wb') as f:
                pickle.dump(splitShorelines, f)
            print('Saved to: ', fp)
            return splitShorelines
        return shorelines

    def split_polygons(self, geometries):
        print('Splitting polygons')
        splitPolys = []
        for polygon in geometries:
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            splitPolys.extend(self.split_polygon(polygon))
        return splitPolys

    def split_polygon(self, geo, cnt=0):
        """Split a Polygon into two parts across it's shortest dimension"""
        (minx, miny, maxx, maxy) = geo.bounds
        width = maxx - minx
        height = maxy - miny
        # if max(width, height) <= threshold or cnt == 250:
        if geo.envelope.area < self.splitThreshold ** 2 or cnt == 250:
            # either the polygon is smaller than the threshold, or the maximum
            # number of recursions has been reached
            return [geo]
        if height >= width:
            # split left to right
            a = box(minx, miny, maxx, miny + height/2)
            b = box(minx, miny + height / 2, maxx, maxy)
        else:
            # split top to bottom
            a = box(minx, miny, minx+width/2, maxy)
            b = box(minx + width / 2, miny, maxx, maxy)
        result = []
        for d in (a, b,):
            c = geo.intersection(d)
            if not isinstance(c, GeometryCollection):
                c = [c]
            for e in c:
                if isinstance(e, (Polygon, MultiPolygon)):
                    result.extend(self.split_polygon(e, cnt+1))
        if cnt > 0:
            return result
        # convert multi part into single part
        finalResult = []
        for g in result:
            if isinstance(g, MultiPolygon):
                finalResult.extend(g)
            else:
                finalResult.append(g)
        return finalResult

    def split_linearring(self, geo, cnt=0):
        """Split a Polygon into two parts across it's shortest dimension"""
        (minx, miny, maxx, maxy) = geo.bounds
        width = maxx - minx
        height = maxy - miny
        # if max(width, height) <= threshold or cnt == 250:
        if geo.envelope.area < self.splitThreshold ** 2 or cnt == 250:
            # either the polygon is smaller than the threshold, or the maximum
            # number of recursions has been reached
            return [geo]
        if height >= width:
            # split left to right
            a = box(minx, miny, maxx, miny + height/2)
            b = box(minx, miny + height / 2, maxx, maxy)
        else:
            # split top to bottom
            a = box(minx, miny, minx+width/2, maxy)
            b = box(minx + width / 2, miny, maxx, maxy)
        result = []
        for d in (a, b,):
            c = geo.intersection(d)
            if not isinstance(c, GeometryCollection):
                c = [c]
            for e in c:
                if isinstance(e, (Polygon, MultiPolygon)):
                    result.extend(self.split_linearring(e, cnt+1))
        if cnt > 0:
            return result
        # convert multi part into single part
        finalResult = []
        for g in result:
            if isinstance(g, MultiPolygon):
                finalResult.extend(g)
            else:
                finalResult.append(g)
        return finalResult

    def get_eca_rtree(self):
        fp = self.DIR / 'data/navigation_area/secas'
        with open(fp, 'rb') as f:
            ecas = pickle.load(f)

        if os.path.exists(fp / '.idx'):
            return {'rtree': Index(fp), 'geometries': ecas}
        else:
            return populate_rtree(ecas, fp)


def populate_rtree(geometries, fp):
    # Populate R-tree index with bounds of geometries
    print('Populate {} tree'.format(fp))
    idx = Index(fp.as_posix())
    for i, geo in enumerate(geometries):
        idx.insert(i, geo.bounds)
    idx.close()

    return {'rtree': Index(fp.as_posix()), 'geometries': geometries}


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords[:-1]):
        pd = line.project(Point(p))
        if pd >= distance:
            return [LineString(coords[:i+1]), LineString(coords[i:])]
    return [LineString(line)]


if __name__ == '__main__':
    import numpy as np

    from matplotlib.collections import PatchCollection
    from matplotlib import patches
    from mpl_toolkits.basemap import Basemap

    DIR = Path('D:/')
    os.chdir('..')
    pars = {'res': 'l', 'splits': 10, 'avoidAntarctic': False, 'avoidArctic': False}
    generator = NavigableAreaGenerator(parameters=pars)
    bathymetry = generator.get_bathymetry_rtree()

    # Plot on Basemap
    _fig, _ax = plt.subplots()
    m = Basemap(projection='robin', lon_0=0, resolution='l', ax=_ax)
    m.drawmapboundary(fill_color='red', zorder=1)
    m.drawcoastlines(color='black')
    m.fillcontinents(color='lightgray', lake_color='lightgray', zorder=2)
    m.readshapefile(DIR / "data/bathymetry_200m/ne_10m_bathymetry_K_200", 'ne_10m_bathymetry_K_200', drawbounds=False)
    ps = [patches.Polygon(np.array(shape), True) for shape in m.ne_10m_bathymetry_K_200]
    _ax.add_collection(PatchCollection(ps, facecolor='white', zorder=2))

    plt.show()

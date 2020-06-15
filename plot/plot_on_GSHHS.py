import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from math import floor, ceil


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


def plot_on_gshhs(individual):
    waypoints = [item[0] for item in individual]
    min_x, min_y = min(waypoints, key=lambda t: t[0]), min(waypoints, key=lambda t: t[1])
    max_x, max_y = max(waypoints, key=lambda t: t[0]), max(waypoints, key=lambda t: t[1])
    m = 5  # margin
    extent = [max(floor(min_x[0]) - m, -180), min(ceil(max_x[0]) + m, 180),
              max(floor(min_y[1]) - m, -90), min(ceil(max_y[1]) + m, 90)]  # [left, right, bottom, top)

    fig, ax = make_map(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot shorelines
    shp = shapereader.Reader('C:/dev/data/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp')
    for record, geometry in zip(shp.records(), shp.geometries()):
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black')

    # Plot waypoints
    waypoints = [item[0] for item in individual]
    edges = zip(waypoints[:-1], waypoints[1:])
    for e in edges:
        plt.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color='blue', linewidth=1, marker='o', markersize=3, transform=ccrs.PlateCarree())


if __name__ == "__main__":
    from deap import base, creator

    # Create Fitness and Indiviul types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    with open('C:/dev/projects/naval_route_planner/output/sorted_population1', 'rb') as f:
        pareto_solutions = pickle.load(f)

    idx = 40
    print([item[1] for item in pareto_solutions[idx]])
    print(pareto_solutions[idx].fitness.values)
    plot_on_gshhs(pareto_solutions[idx])
    plt.show()

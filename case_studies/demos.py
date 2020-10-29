import matplotlib.pyplot as plt
import numpy as np


def create_currents(nDays=1, returnDict=False, plot=False):
    lons0 = np.linspace(-179.875, 179.875, 1440)
    lats0 = np.linspace(-89.875, 89.875, 720)

    # Get subset
    lons = lons0[(lons0 > -5) & (lons0 < 5)]
    lats = lats0[(lats0 > -2.5) & (lats0 < 2.5)]

    xx0, yy0 = np.meshgrid(lons0, lats0)
    xx, yy = np.meshgrid(lons, lats)
    u = 2 * np.cos(yy*np.pi/3 + np.pi)
    v = np.zeros(np.shape(u))
    u0 = 2 * np.cos(yy0 * np.pi/3 + np.pi)
    v0 = np.zeros(np.shape(u0))

    if plot:
        Q = plt.quiver(lons, lats, u, v, u, units='x', pivot='mid', width=0.02, cmap='bwr', scale=10)
        plt.quiverkey(Q, 0.9, 0.9, 1, r'$1$ knot', labelpos='E', coordinates='figure')
    if returnDict:
        return {'u': u0, 'v': v0, 'lons': lons0, 'lats': lats0}
    else:
        arr = np.empty([2, nDays * 8, len(lats0), len(lons0)])
        for i, current in enumerate([u0, v0]):
            for j in range(nDays):
                arr[i, j] = current
        return arr


def create_wind(plot=False):

    lons = np.linspace(-180, 179.5, 720)
    lats = np.linspace(-90, 90, 361)

    x, y = np.meshgrid(lons, lats)
    BN = np.round(6 * (np.cos(y * np.pi / 12 + np.pi)+1))
    TWD = np.zeros(np.shape(BN))

    if plot:
        plt.contourf(lons, lats, BN)
        plt.colorbar()


if __name__ == '__main__':
    import main
    import pickle

    def test_demo():
        par = {'delFactor': 1.2,
               'graphDens': 8,       # Recursion level graph
               'graphVarDens': 2}    # Variable recursion level graph

        planner = main.RoutePlanner(inputParameters=par, criteria={'minimalTime': True, 'minimalCost': True})

        rawIn = planner.compute(((-10, 0), (10, 0)), startDate=None, recompute=True, current=True)
        proc, raw = planner.post_process(rawIn)

        with open('D:/output/demoRaw', 'wb') as fh:
            pickle.dump((proc, raw), fh)


    test_demo()
    # plt.show()

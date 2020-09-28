import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

for res in ['c', 'l', 'i', 'h', 'f']:
    m = Basemap(projection='merc', resolution=res,
                llcrnrlat=50.667807, llcrnrlon=2.59,
                urcrnrlat=53.740532, urcrnrlon=7.758459)
    m.drawcoastlines()
    m.fillcontinents(color='lightgray')
    m.drawmapboundary()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('D:/output/NL_resolution_{}.pdf'.format(res), bbox_inches='tight', pad_inches=0)
    plt.close('all')
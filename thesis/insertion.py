import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import random

from math import sqrt, cos, sin, pi, degrees, atan2
from matplotlib import patches
from pathlib import Path

DIR = Path('D:/')

rcParams["font.serif"] = ["TeX Gyre Pagella"]
# rcParams["font.family"] = ["serif"]

widthRatio = 0.5

p1, p2 = np.array([-1, 0]), np.array([1, 0])
ctr = (p1 + p2) / 2  # Edge centre
width = np.linalg.norm(p1 - p2)  # Ellipse width
height = width * widthRatio  # Ellipse height
print('w', width, 'h', height)

random.seed(3)


def plot_uniform_shapes(n, bins, save=False):
    def rhombus(nPoints):
        xx, yy = [], []
        for _ in range(nPoints):
            a = width / 2.
            b = height / 2.

            v1 = np.array([a, - b])
            v2 = np.array([a, b])
            offset = np.array([a, 0.])
            ptOrigin = random.random() * v1 + random.random() * v2 - offset

            # Transform
            dx, dy = p2 - p1
            theta = np.arctan2(dy, dx)
            cosTh = np.cos(theta)
            sinTh = np.sin(theta)
            rotation_matrix = np.array([[cosTh, -sinTh],
                                        [sinTh, cosTh]])
            ptRotated = np.dot(rotation_matrix, ptOrigin)  # Rotate
            x, y = ctr + ptRotated  # Translate
            x -= np.int(x / 180) * 360
            y = np.clip(y, -90, 90)

            xx.append(x)
            yy.append(y)
        return xx, yy

    def ellipse_uniform(nPoints):
        xx, yy = [], []
        for _ in range(nPoints):
            # Generate a random point inside a circle of radius 1
            rho = random.random()
            phi = random.random() * 2 * pi
            xy = sqrt(rho) * np.array([cos(phi), sin(phi)])

            ptOrigin = xy * np.array([width, height]) / 2.0

            # Transform
            dy, dx = p2[1] - p1[1], p2[0] - p1[0]
            if dx != 0:
                slope = dy / dx
                rotX = 1 / sqrt(1 + slope ** 2)
                rotY = slope * rotX
            else:
                rotX = 0
                rotY = 1

            cosA, sinA = rotX, rotY
            rotation_matrix = np.array(((cosA, -sinA), (sinA, cosA)))
            ptRotated = np.dot(rotation_matrix, ptOrigin)  # Rotate
            x, y = ctr + ptRotated  # Translate
            x -= np.int(x / 180) * 360
            y = np.clip(y, -90, 90)

            xx.append(x)
            yy.append(y)
        return xx, yy

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='row')

    rx, ry = rhombus(n)
    ex, ey = ellipse_uniform(n)

    ax1.hist(rx, bins=bins, color='black')
    ax2.hist(ex, bins=bins, color='black')
    ax3.scatter(rx, ry, s=.5, color='black')
    ax4.scatter(ex, ey, s=.5, color='black')

    vx = [p1[0], ctr[0], p2[0], ctr[0]]
    vy = [p1[1], -.5 * height, p2[1], .5 * height]

    rhombus = patches.Polygon(xy=list(zip(vx, vy)), fill=False, ec='red')
    ax3.add_patch(rhombus)

    for ax in [ax1, ax2]:
        # ax.set_axis_off()
        # ax.xaxis.set_major_locator(plt.NullLocator())
        # ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_ylim(0, 80)
        ax.axes.get_xaxis().set_visible(False)

    ax2.axes.get_yaxis().set_ticklabels([])

    ellipse = patches.Ellipse(tuple(ctr), width=width, height=height,
                              angle=degrees(atan2(p2[1] - p1[1], p2[0] - p1[0])),
                              fill=False,
                              ec='red')
    ax4.add_patch(ellipse)

    # fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    for ax in [ax3, ax4]:
        ax.set_axis_off()
        ax.set_ylim(-width / 2, width / 2)
        ax.set_xlim(p1[0], p2[0])
        ax.axis('equal')
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=10, color='blue', zorder=2)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    if save:
        fig.savefig(DIR / "output/figures/OPS_rhombus_ellipse.pdf", bbox_inches='tight', pad_inches=0)

    return fig, (ax1, ax2, ax3, ax4)


def plot_gaussian_shape(n, plot=True, save=False):
    def ellipse_gauss(nPoints):
        xx, yy = [], []
        for _ in range(nPoints):
            mu, cov = [0, 0], np.array([[1, 0], [0, 1]]) * 1/5.991
            xy = np.random.multivariate_normal(mu, cov, 1).squeeze()

            ptOrigin = xy * np.array([width, height]) / 2.0

            # Transform
            dy, dx = p2[1] - p1[1], p2[0] - p1[0]
            if dx != 0:
                slope = dy / dx
                rotX = 1 / sqrt(1 + slope ** 2)
                rotY = slope * rotX
            else:
                rotX = 0
                rotY = 1

            cosA, sinA = rotX, rotY
            rotation_matrix = np.array(((cosA, -sinA), (sinA, cosA)))
            ptRotated = np.dot(rotation_matrix, ptOrigin)  # Rotate
            x, y = ctr + ptRotated  # Translate
            x -= np.int(x / 180) * 360
            y = np.clip(y, -90, 90)

            xx.append(x)
            yy.append(y)
        return xx, yy

    gx, gy = ellipse_gauss(n)

    count = 0
    for pt in zip(gx, gy):
        x, y = pt
        if (x/(width/2)) ** 2 + (y/(height/2)) ** 2 <= 1:
            count += 1
    ellipse = patches.Ellipse(tuple(ctr), width, height, angle=degrees(atan2(p2[1] - p1[1], p2[0] - p1[0])), fill=False,
                              ec='red', linewidth=2)
    contained = sum(ellipse.contains_points(list(zip(gx, gy))))

    print(contained / n)
    print(count / n)
    print(count, n - count, n)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(gx, gy, s=1, color='black')

        ellipse = patches.Ellipse(tuple(ctr), width, height,
                                  angle=degrees(atan2(p2[1] - p1[1], p2[0] - p1[0])),
                                  fill=False,
                                  ec='red',
                                  linewidth=2)

        ax.add_patch(ellipse)

        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.set_axis_off()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_ylim(-width / 2, width / 2)
        ax.set_xlim(p1[0], p2[0])
        ax.axis('equal')
        # ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=10, color='blue', zorder=2)

        if save:
            fig.savefig(DIR / "output/figures/OPS_gauss1.pdf", bbox_inches='tight', pad_inches=0)

        return fig, ax


N = 1000
BINS = 30

plot_uniform_shapes(N, BINS, save=True)

plt.show()

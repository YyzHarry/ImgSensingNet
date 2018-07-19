# -*- coding: utf-8 -*-

import numpy as np
import time
import json
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('text', usetex=True)


# Total_Time_Stamps, Total_Sensor_Num
Total_Time_Stamps = 30
Total_Sensor_Num = 30


# 2D Region Establish -- 63*44 (20m*20m)
Xcoord = 630
Ycoord = 440
FigArray = [[0 for j in range(Ycoord)] for i in range(Xcoord)]


# ========================================== Read in Sensor Locations
file = open('./LocatCoordinates.txt', 'r')
data = file.readlines()

# X coordinates
odom = data[0].split()
num = map(int, odom)
spotX = (list(num))
print(spotX)

# Y coordinates
odom = data[1].split()
num = map(int, odom)
spotY = (list(num))
print(spotY)

# ========================================== Fig1: All points
# Fig-Establish & Name
plt.figure('Simplified sensor distributions')
ax = plt.gca()
# x & y label
# ax.set_xlabel('Length (m)')
# ax.set_ylabel('Width (m)')
# Move origin point to upper-left
ax.set_xlim(left=0, right=Ycoord)
ax.set_ylim(bottom=Xcoord, top=0)
# Set x&y axis w/ equal min-interval
ax.set_aspect(1)
# Turn off s&y axis
plt.xticks([])
plt.yticks([])

# Scatter Fig
# c - color; s - square; alpha - transmission
ax.scatter(spotY, spotX, c='black', s=100, alpha=0.5)

plt.annotate('$\mathbf{Sensor}$', xy=(230, 300), xycoords='data', xytext=(+30, +30),
             textcoords='offset points', fontsize=20, fontweight='heavy',
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.3"))

plt.show()

# ========================================== Fig2: Point-of-Interests
# Point-of-Interest (5)
poiX = [160, 300, 60, 490, 510]
poiY = [10, 220, 370, 60, 360]

# Fig-Establish & Name
plt.figure('Extract Point-of-Interests')
ax = plt.gca()
# Move origin point to upper-left
ax.set_xlim(left=0, right=Ycoord)
ax.set_ylim(bottom=Xcoord, top=0)
# Set x&y axis w/ equal min-interval
ax.set_aspect(1)
# Turn off s&y axis
plt.xticks([])
plt.yticks([])

# Scatter Fig
# c - color; s - square; alpha - transmission
ax.scatter(spotY, spotX, c='black', s=100, alpha=0.5)
ax.scatter(poiY, poiX, c='red', s=200, alpha=1)

plt.annotate('$\mathbf{Point}\ \mathbf{of}\ \mathbf{interest}$', xy=(235, 300), xycoords='data', xytext=(-50, +30),
             textcoords='offset points', fontsize=20, fontweight='heavy',
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.3"))

plt.show()


# ========================================== K-means Clustering
label = [0 for i in range(Total_Sensor_Num)]

for i in range(Total_Sensor_Num):
    dist_min = 0xffff
    for j in range(5):
        dist = (spotX[i]-poiX[j])**2 + (spotY[i]-poiY[j])**2
        if dist < dist_min:
            dist_min = dist
            label[i] = j

# Divide different labels
grp1X = []
grp1Y = []
grp2X = []
grp2Y = []
grp3X = []
grp3Y = []
grp4X = []
grp4Y = []
grp5X = []
grp5Y = []

for i in range(Total_Sensor_Num):
    if label[i] == 0:
        grp1X.append(spotX[i])
        grp1Y.append(spotY[i])
    if label[i] == 1:
        grp2X.append(spotX[i])
        grp2Y.append(spotY[i])
    if label[i] == 2:
        grp3X.append(spotX[i])
        grp3Y.append(spotY[i])
    if label[i] == 3:
        grp4X.append(spotX[i])
        grp4Y.append(spotY[i])
    if label[i] == 4:
        grp5X.append(spotX[i])
        grp5Y.append(spotY[i])


# ========================================== Fig3: Clustering
# Fig-Establish & Name
plt.figure('Clustering')
ax = plt.gca()
# Move origin point to upper-left
ax.set_xlim(left=0, right=Ycoord)
ax.set_ylim(bottom=Xcoord, top=0)
# Set x&y axis w/ equal min-interval
ax.set_aspect(1)
# Turn off s&y axis
plt.xticks([])
plt.yticks([])

# Scatter Fig
# c - color; s - square; alpha - transmission
ax.scatter(poiY[0], poiX[0], c='plum', s=200, alpha=1)
ax.scatter(grp1Y, grp1X, c='plum', s=100, alpha=1)
ax.scatter(poiY[1], poiX[1], c='deepskyblue', s=200, alpha=1)
ax.scatter(grp2Y, grp2X, c='deepskyblue', s=100, alpha=1)
ax.scatter(poiY[2], poiX[2], c='limegreen', s=200, alpha=1)
ax.scatter(grp3Y, grp3X, c='limegreen', s=100, alpha=1)
ax.scatter(poiY[3], poiX[3], c='gold', s=200, alpha=1)
ax.scatter(grp4Y, grp4X, c='gold', s=100, alpha=1)
ax.scatter(poiY[4], poiX[4], c='orangered', s=200, alpha=1)
ax.scatter(grp5Y, grp5X, c='orangered', s=100, alpha=1)

circle = plt.Circle((355, 120), 80, color='limegreen', linestyle='--', linewidth=3, fill=False)
ax.add_artist(circle)
'''
plt.text(180, 40, '$\mathbf{Clustering}$',
         fontdict={'size': 20, 'color': 'limegreen'})
'''
plt.annotate('$\mathbf{Clustering}$', xy=(270, 100), xycoords='data', xytext=(-100, +30),
             textcoords='offset points', fontsize=20,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.3"))

plt.show()


# ========================================== Multi-Sites Weighted Voronoi
weights = [len(grp1X), len(grp2X), len(grp3X), len(grp4X), len(grp5X)]

mean_grp1X = sum(grp1X)/len(grp1X)
mean_grp1Y = sum(grp1Y)/len(grp1X)
mean_grp2X = sum(grp2X)/len(grp2X)
mean_grp2Y = sum(grp2Y)/len(grp2X)
mean_grp3X = sum(grp3X)/len(grp3X)
mean_grp3Y = sum(grp3Y)/len(grp3X)
mean_grp4X = sum(grp4X)/len(grp4X)
mean_grp4Y = sum(grp4Y)/len(grp4X)
mean_grp5X = sum(grp5X)/len(grp5X)
mean_grp5Y = sum(grp5Y)/len(grp5X)

mean_grpX = [mean_grp1X, mean_grp2X, mean_grp3X, mean_grp4X, mean_grp5X]
mean_grpY = [mean_grp1Y, mean_grp2Y, mean_grp3Y, mean_grp4Y, mean_grp5Y]

'''
plt.figure('Clustering')
ax = plt.gca()
# x & y label
ax.set_xlabel('Length (m)')
ax.set_ylabel('Width (m)')
# Move origin point to upper-left
ax.set_xlim(left=0, right=Ycoord)
ax.set_ylim(bottom=Xcoord, top=0)
# Set x&y axis w/ equal min-interval
ax.set_aspect(1)

ax.scatter(poiY[0], poiX[0], c='r', s=100, alpha=1)
ax.scatter(grp1Y, grp1X, c='r', s=100, alpha=1)
ax.scatter(poiY[1], poiX[1], c='b', s=100, alpha=1)
ax.scatter(grp2Y, grp2X, c='b', s=100, alpha=1)
ax.scatter(poiY[2], poiX[2], c='g', s=100, alpha=1)
ax.scatter(grp3Y, grp3X, c='g', s=100, alpha=1)
ax.scatter(poiY[3], poiX[3], c='c', s=100, alpha=1)
ax.scatter(grp4Y, grp4X, c='c', s=100, alpha=1)
ax.scatter(poiY[4], poiX[4], c='m', s=100, alpha=1)
ax.scatter(grp5Y, grp5X, c='m', s=100, alpha=1)
ax.scatter(mean_grpY, mean_grpX, c='k', s=100, alpha=1)
plt.show()
'''


# ========================================== Fig4: Weighted Voronoi
for i in range(Xcoord):
    for j in range(Ycoord):
        dist_min = 0xffff
        for k in range(5):
            dist = ((i-mean_grpX[k])**2 + (j-mean_grpY[k])**2) / math.sqrt(weights[k]-1)
            if dist < dist_min:
                dist_min = dist
                FigArray[i][j] = k


colors = ['plum', 'deepskyblue', 'limegreen', 'gold', 'orangered']
bounds = [i for i in range(6)]

cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Fig-Establish & Name
plt.figure('Weighted Voronoi')
ax = plt.gca()
# Move origin point to upper-left
ax.set_xlim(left=0, right=Ycoord)
ax.set_ylim(bottom=Xcoord, top=0)
# Set x&y axis w/ equal min-interval
ax.set_aspect(1)
# Turn off s&y axis
plt.xticks([])
plt.yticks([])

plt.imshow(FigArray, interpolation='none', cmap=cmap, norm=norm)

ax.scatter(spotY, spotX, c='black', s=100, alpha=0.5)
ax.scatter(mean_grpY, mean_grpX, c='white', s=200, alpha=0.9, marker='*')

plt.annotate('$\mathbf{Center}\ \mathbf{of}\ \mathbf{region}$', xy=(252, 300), xycoords='data', xytext=(-65, +30),
             textcoords='offset points', fontsize=20,
             arrowprops=dict(facecolor='white', arrowstyle='->', connectionstyle="arc3,rad=-0.3"))

plt.show()

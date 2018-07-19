# -*- coding: utf-8 -*-

# import scipy

import numpy as np
import pymysql
import time
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model


# Total_Time_Stamps, Total_Sensor_Num
Total_Time_Stamps = 30
Total_Sensor_Num = 30

# Connect to Database
config = {
        'host': '59.110.46.30',
        'user': 'bdm254844563',
        'password': 'mysqlsjk',
        'port': 3306,
        'database': 'bdm254844563_db',
        'charset' : 'utf8'
}
conn = pymysql.connect(**config)
cur = conn.cursor()

# Origin Point
originalPoint = [116.311863, 40.003892]
# lastPoint = [116.321834,40.003892]

# Convert lat & lng to meters
indexX = 85609.08
indexY = 111293.97

# Data point
IDNum = []
spotX = []
spotY = []
locationName = []
pm25 = []
pmIndex = []
dataInf = []

# Get current time
nowtime = datetime.datetime.now() + datetime.timedelta(minutes=-datetime.datetime.now().minute)

# 2D Region Establish -- 63*43 (20m*20m)
Xnum = 43
Ynum = 63
array = [[0 for i in range(Xnum)] for j in range(Ynum)]
count = [1 for i in range(Total_Sensor_Num)]

# Find ID-outside
cur.execute("SELECT ID,location_des,location_coo,pm_index FROM device_global_inf WHERE city = 0 AND ID > 100")
result = cur.fetchall()
i = 0
for row in result:
    i += 1
    IDNum.append(row[0])
    locationName.append(row[1])
    pmIndex.append(row[3])
    split = row[2].index(',')
    x = float(row[2][0:split])
    y = float(row[2][split+1:])
    spotX.append(math.floor(abs(x-originalPoint[0]) * indexX / 20))
    spotY.append(math.floor(abs(y-originalPoint[1]) * indexY / 20))
    # label the point
    # print(spotX[-1],spotY[-1])
    print(spotY[-1]*10, spotX[-1]*10)

    array[spotY[-1]][spotX[-1]] = i

# Sensor locations
with open('./LocatCoordinates.txt', 'w') as f:
    # f.write('[\n')
    for i in range(Total_Sensor_Num):
        f.write(str(spotY[i]*10))
        if i == (Total_Sensor_Num - 1):
            f.write('\n')
        else:
            f.write(' ')
    for i in range(Total_Sensor_Num):
        f.write(str(spotX[i]*10))
        if i == (Total_Sensor_Num - 1):
            f.write('\n')
        else:
            f.write(' ')


tmp = 0
for i in range(Xnum):
    for j in range(Ynum):
        disSqr = 3000
        area = -1
        for k in range(len(spotX)):
            tmpDisSqr = (i-spotX[k])**2 + (j-spotY[k])**2
            if tmpDisSqr < disSqr:
                disSqr = tmpDisSqr
                area = k
        array[j][i] = array[spotY[area]][spotX[area]]
        count[area] = count[area]+1
for k in range(len(spotX)):
    array[spotY[k]][spotX[k]] = 0

colors = ['red','#99FFFF','#33CCCC','#00CC99','#99FF99','#009966','#33FF33','#33FF00','#99CC33','#66CCCC',	'#99FFCC',	'#99CC99',	'#CCFFCC',	'#66CC33',	'#339900',	'#999900',	'#CCCC00',	'#669900','#66FFCC','#66FF66','#009933','#00CC33','#66FF00','#336600','#339933',	'#33FF66',	'#33CC33',	'#99FF00',	'#669900',	'#666600',	'#00FFFF']

bounds = [i for i in range(30)]


cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

plt.imshow(array, interpolation='none', cmap=cmap, norm=norm)
plt.show()

a = np.loadtxt('derivation.json')
drv = a.transpose()

print(a.shape)
total_drv = [0 for i in range(Total_Sensor_Num + 1)]

for i in range(Xnum):
    for j in range(Ynum):
        if array[j][i] == 0:
            total_drv[array[j-1][i-1]] = total_drv[array[j-1][i-1]] + drv[j][i]
        else:
            total_drv[array[j][i]] = total_drv[array[j][i]] + drv[j][i]
print(total_drv)
print(count)
avg_drv = [0 for i in range(Total_Sensor_Num+1)]
for i in range(0, Total_Sensor_Num):
    avg_drv[i] = total_drv[i+1]/count[i]
for i in range(0, Total_Sensor_Num):
    print(locationName[i], i+1, count[i], '% .2f' % avg_drv[i])

cur.close()
conn.close()

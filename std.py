from random import random
import math

def rand_cluster(n,r,c1,c2):
    """returns n random points in disk of radius r centered at c"""
    x,y = c1,c2
    points = []
    for i in range(n):
        theta = 2*math.pi*random()
        s = r*random()
        points.append((x+s*math.cos(theta), y+s*math.sin(theta)))
    return points

'''x1 = np.transpose(rand_cluster(1000,0.3,0.75,0.75))
x2 = np.transpose(rand_cluster(1000,0.3,0.75,0.25))
x3 = np.transpose(rand_cluster(1000,0.3,0.25,0.75))
x4 = np.transpose(rand_cluster(1000,0.,0.25,0.25))
plt.scatter(x1[0],x1[1])
plt.scatter(x2[0],x2[1])
plt.scatter(x3[0],x3[1])
plt.scatter(x4[0],x4[1])
plt.show()'''

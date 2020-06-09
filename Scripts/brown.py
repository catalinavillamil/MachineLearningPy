# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 09:40:03 2017

@author: Usuario
"""

import numpy as np
from pylab import plot, show, grid, xlabel, ylabel,title,axis
import matplotlib.pyplot as plt
from scipy.stats import norm


def brownian(x0, n, dt, delta, out=None):
    global x_cant
    x0 = np.asarray(x0)


    r = norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))
    x_cant=np.random.randint(10,100,n)
    if out is None:
        out = np.empty(r.shape)

 
    np.cumsum(r, axis=-1, out=out)

    out += np.expand_dims(x0, axis=-1)

    return out
    
# Proceso de weiner.
delta = 5
delta1 = 15
# Tiempo total de simulación.
T = 50.0
T1 = 50.0
# Número de pasos.
N = 1500
N1 = 1500
# Tamaño de pasos
dt = T/N
dt1 = T/N
# Número de realizaciones generadas.
#m = 20
x_cantidad=np.empty((2,N+1))
x_cantidad[0,0]=5000

x = np.empty((2,N+1))
x[:, 0] = 0.0
x1 = np.empty((2,N+1))
x1[:, 0] = 10.0
# Initial values of x.
#

brownian(x[:,0], N, dt, delta, out=x[:,1:])

plt.subplot(2,2,1)
plot(x[0],x[1])


plot(x[0,0],x[1,0], 'go')
plot(x[0,-1], x[1,-1], 'ro')
title('Nodo_1')
xlabel('x', fontsize=16)
ylabel('y', fontsize=16)
axis('equal')
grid(True)

brownian(x1[:,0], N1, dt1, delta1, out=x1[:,1:])

plt.subplot(2,2,2)
plot(x1[0],x1[1])
plot(x1[0,0],x1[1,0], 'go')
plot(x1[0,-1], x1[1,-1], 'ro')

#t = np.linspace(0.0, N*dt, N+1)
#for k in range(m):
#    plot(t, x[k],'bo')
#xlabel('t', fontsize=16)
#ylabel('x', fontsize=16)
#grid(True)
#show()
title('Nodo_2')
xlabel('x', fontsize=16)
ylabel('y', fontsize=16)
axis('equal')
grid(True)
show()
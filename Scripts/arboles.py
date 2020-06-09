# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:16:55 2017

@author: Usuario
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def entropia(v):
  global arbol
  global arbol1
  global p
  global s
  p=0
  r=0
  s=0
  for n in v:
      p=arbol.node[n]['costo']+p
      print (p)
  for n in v:
      s=((arbol.node[n]['costo']/np.double(p))*(np.log(arbol.node[n]['costo']/np.double(p)))) + s
  s=-1*s
  return s    
nodos=10
conexiones=([0,1],[1,2],[1,3],[1,4],[3,5],[3,6],[2,7],[2,8],[2,9])
arbol=nx.Graph()
arbol.add_nodes_from(np.arange(0,nodos,1))


for i in range(nodos):
###costo=list(np.random.uniform(6000,50000,nodos))
###costo1=list(np.random.uniform(50000,500000,nodos))
 arbol.node[i]['costo']=np.random.uniform(50000,500000)
arbol.add_edges_from(conexiones)
#arbol1.add_edges_from(conexiones)
vecinos=nx.neighbors(arbol,1)
print(entropia(vecinos))
pos = nx.random_layout(arbol)
#pos = nx.random_layout(G)

# rendering
plt.figure(1)
plt.axis('off')
nx.draw_networkx_nodes(arbol, pos)
nx.draw_networkx_edges(arbol, pos, width=3.0)

plt.show()
#plt.subplot(212); plt.axis('off')
#
## rendering
#nx.draw_networkx_nodes(G, pos)
#nx.draw_networkx_edges(G, pos, edge_color=edgewidth)


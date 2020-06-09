# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:59:48 2017

@author: rrenteria
"""

import pandas as pd
import networkx as nx
import numpy as np
import sklearn as sk
from collections import defaultdict
import matplotlib.pyplot as plt
import math
#import seaborn as sns
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize
def grafica_comunidades():
    global coherencia_comunidad
    global cli2
    global comun
    c=list()
    comuni=list()
    posi_clique=list()
    escala=np.zeros((comun,len(cli2)))
    for i in range(len(coherencia_comunidad)):  
       a,b=coherencia_comunidad[i]
       escala[b,i]=a
       posi_clique.append(i)
       comuni.append(b)
       c.append(a)
    data={'coherence':c,'community':comuni,'clique':posi_clique}
    data=pd.DataFrame(data)
    data=data[data['coherence']>0]
    sns.jointplot("clique","coherence", kind="kde", data=data,lim={0,10,0,10})#,color=b)#kind="kde", data=data)#, color="#8080C0") 
    sns.jointplot("community","coherence", kind="kde", data=data,lim={0,10,0,10})
def buscar_arista(a,b):
    global arista_comunidades
    sal=0
    for i in range(len(arista_comunidades)):
        a1,b1=arista_comunidades[i]
        if (a1==a and b1==b)or(b==a1 and a==b1):
          sal=1
          break
    return sal
  
def crear_comunidades():
   global cli2
   global arista_comunidades
   global source_comunidades
   global target_comunidades
   global pesos_comunidades
   global orden_comunidad
   global nx
   global red
   global coherencia_comunidad
   global comun
   coherencia_comunidad=list()
   mx=nx.adjacency_matrix(red).todense()
   source_comunidades=list()
   target_comunidades=list()
   pesos_comunidades=list()
   arista_comunidades=list()
   orden_comunidad=list()
   comun=0
   for i in range(len(k_comunidades)):
       coherencia_comunidad.append([coherencia_clique[i],comun])
       for n in range(len(k_comunidades)):
           if k_comunidades[i,n]==1:
               
            nodos=cli2[i]
            a=nodos[0]
            if len(nodos)>1:
              for j in range(1,len(nodos)):
               if buscar_arista(a,nodos[j])==0:
                arista_comunidades.append([a,nodos[j]])
                source_comunidades.append(a)
                target_comunidades.append(nodos[j])
                pesos_comunidades.append(mx[a,nodos[j]])
                orden_comunidad.append(comun)
                a=nodos[j]
                
           else:
               
              comun=comun+1
   guardar_red_gephi(source_comunidades,target_comunidades,'Diagnostico','Undirected',pesos_comunidades,"nodos_overlapping_comunidades_cliques","red_overlapping_comunidades")           
def adjust_cliques(k):
    global cli2
    l=len(cli2)
    i=0
    while i<l:
         dato=len(cli2[i])
         if dato<k:
             cli2.pop(i)
             l=(len(cli2))
         i=i+1
        
def intensidad_clique_cortada(k):
    global cli2
    global filtrados
    global red1
    global nx
    global i_clique
    global coherencia_clique
    enlances=np.zeros((len(filtrados)))
    mx=nx.adjacency_matrix(red).todense()
    i_clique=np.zeros((len(filtrados)))
    coherencia_clique=np.zeros((len(filtrados)))
    for i in range(len(filtrados)):
        
         datos=cli2[filtrados.iloc[i]]
         t=len(datos)
         if t>=k:
             l=0
             link=0
             d=0
             multi=1.0
             coherencia=0
             while t>l:
               
               for index in range(l,(t-1)):
                   print(i,index)
                   d=(mx[datos[index],datos[index+1]])
                   multi=multi*d
                   coherencia=d+coherencia
                   link=link+1
               l=l+1    
             enlances[i]=link  
             i_clique[i]=pow(d,(1.0/link))
             coherencia=(coherencia/link)
             coherencia_clique[i]=i_clique[i]/coherencia
    
    
    
def deteccion_comunidades_overlap_cliques(k):
    global k_comunidades
    for i in range(len(k_comunidades)):

       for j in range(len(k_comunidades)):
        if i!=j:    
         if k_comunidades[i,j]>=(k-1) :
            k_comunidades[i,j]=1 
         else: 
           k_comunidades[i,j]=0  
        else:
         if k_comunidades[i,i] >= k:
            k_comunidades[i,i]=1 
         else:
           k_comunidades[i,i]=0
    
           
def deteccion_overlap_cliques():
    global cli2
    global k_comunidades
    global k_comunidades_1
    k_comunidades=np.zeros((len(cli2),len(cli2)))
    r=list()
    s=list()
    for i in range(len(cli2)):
       #if k>=len(cli[i]): 
        for j in range(i+1,len(cli2)) :
         r=cli2[i]
         s=cli2[j]
         d=0
         for n in r:
             #print(n)
             for m in s:
                 #print(m)
                 if m==m:
                     #print('ok')
                     d=d+1            
         k_comunidades[i,j]=d
         k_comunidades[j,i]=d
         #np.max(k_comunidades)
        k_comunidades[i,i]=len(cli2[i])
      
def max_flujo_clique():
    global pila
    global i_clique
    global cli2
    global cliques_fuertes
    global coherencia_clique
    cliques_fuertes=list()
    indices=np.where(coherencia_clique==coherencia_clique.max())
    for i in range(2):
      indice=np.where(i_clique==i_clique.max())
      pmax=indice[0][0]
      cliques_fuertes.append(pmax)
      i_clique[pmax]=0
      
def intensidad_clique():
    global cli2
    global nx
    global red
    global i_clique
    global coherencia_clique
    global enlances
    global pesos
    enlances=np.zeros((len(cli2)))
    k=3
    mx=nx.adjacency_matrix(red).todense()
    i_clique=np.zeros((len(cli)))
    coherencia_clique=np.zeros((len(cli)))
    for i in range(len(cli)):
      if len(cli[i])>=k:
         datos=cli[i]
         t=len(datos)
         l=0
         link=0
         d=0
         multi=1.0
         coherencia=0
         while t>l:
           for index in range(l,(t-1)):
               d=(mx[datos[index],datos[index+1]])
               multi=multi*d
               coherencia=d+coherencia
               link=link+1
           l=l+1    
         enlances[i]=link  
         i_clique[i]=pow(d,(1.0/link))/sum(pesos) 
         coherencia=(coherencia/link)/sum(pesos)
         coherencia_clique[i]=i_clique[i]/coherencia
def fitting_cola():
    global grados_1
    global grado2
    global punto2_x2
    global a1
    global a2
    global b1
    global b2
    global punto_x
    global punto_x2
#    punto_x=100
#    punto_x2=731
    x=np.arange(punto_x,len(grados_1),1)
    x2=np.arange(punto_x2,punto2_x2)
    y=grados_1[punto_x:len(grados_1)]
    y1=grado2[punto_x2:punto2_x2]
    a1,b1= optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y,  p0=(punto_x, grados_1[punto_x]))
    a2,b2= optimize.curve_fit(lambda t1,e,d: e*np.exp(d*t1),  x2,  y1,  p0=(punto_x2, grado2[punto_x2]))


def corte_hub(G,cant_hubs):
   global aristas
   global aristas_cortadas1
   global pesos_cortados1
   global pesos
   aristas_cortadas1=aristas.copy()
   pesos_cortados1=pesos.copy()
   datos=list()
   hubs=list()
   grados=G.degree().values()
   for dato in grados:datos.append(dato) 
   
   for i in range(cant_hubs):     
     j=datos.index(max(datos))
     hubs.append(j)
     datos[j]=0
   
   p=len(aristas_cortadas1)
   n=0    
   while p>n:     
       s,t=aristas_cortadas1[n]
       for i in range(cant_hubs): 
        if (s==pila[hubs[i]] or t==pila[hubs[i]] ):
            aristas_cortadas1.remove([s,t])
            pesos_cortados1.pop(n)
            n=0
            p=len(aristas_cortadas1)
           
      
        n=n+1   
      
def entropia_valor_propio(valores):
    global entropia_red
#    items = sorted ( valores.items () )
#    x, x1 = np.array(items).T
    #plt.axis([np.min(valores),np.max(valores),np.min(entropia_red),np.max(entropia_red)])
    
    plt.grid(True)
    plt.xlabel(r'$ CE $', fontsize=20)
    plt.ylabel(r'$ S $', fontsize=20)
    plt.loglog(valores,entropia_red,'bo')
    plt.show()
def peso_promedio(G):
    global nx
    global limite_corte
    global mx
    mx=nx.adjacency_matrix(G).todense()
    pro_wij=np.zeros((G.number_of_nodes()))
    result=np.zeros((G.number_of_nodes(),G.number_of_nodes()))
    for i in range(G.number_of_nodes()):
              
        result[i,:]=mx[i,:]/np.sum(mx[i,:])
        pro_wij[i]=np.sum(mx[i,:])
    return result,pro_wij 

def entropia(grados):
 
    global entropia_red
    for i in range(len(grados)):
        ent_p=0
        for j in range(len(grados)):
           if grados[i,j]!=0:
                ent_p=(grados[i,j]*np.log(grados[i,j]))+ent_p
           entropia_red[i]=(-1)*ent_p     

def cortar_red():
    global limite_corte
    global pesos
    global aristas
    global aristas_cortadas
    global pesos_cortados

    for i in range(len(pesos)):
        if pesos[i]>limite_corte:
            aristas_cortadas.append(aristas[i])
            pesos_cortados.append(pesos[i])
def cortar_red_1():
    global limite_corte
    global aristas
    global pesos
    global aristas_cortadas
    global pesos_cortados
    global entropia_red
    global prom_wij
    global limite_corte_2
    aristas_cortadas=aristas.copy()
    pesos_cortados=pesos.copy()
    corte=np.mean(entropia_red/prom_wij)
    for i in range(len(pila)):
        if (entropia_red[i]/prom_wij[i])>corte and prom_wij[i]<limite_corte_2:
            print(i)
            n=0
            p=len(aristas_cortadas)
            while n<p:
                s,d=aristas_cortadas[n]
                if s==pila[i] or d==pila[i]:
                    aristas_cortadas.pop(n)
                    pesos_cortados.pop(n)  
                    p=len(aristas_cortadas)
                n=n+1    
        
def plotDegreeDistribution(s,G):  
    global nx
    global degs
    global items
    global x
    global y
    global grados_1
    global xdata
    y1 = np.zeros((G.number_of_nodes()))
    for i in range(G.number_of_nodes()):    
       y1[i] = G.degree(i)
#    y1= (y1*1.0)/sum(y1)
#    y1=np.sort(y1)
#    y=np.flipud(y1)
    #f, ax=plt.subplot(2,sharex=True)
   
    (f, d) = np.histogram(y1, bins=np.linspace(0,np.amax(y1+1),num=np.amax(y1)+2))
    xdata = d[1:len(d)]
    ydata = np.double(np.flipud(np.cumsum(np.flipud(f))))/sum(f) 
    grados_1=ydata
    
    plt.ylabel(r'$P_K$',fontsize = 20)
    plt.xlabel(r'$K/W_k$',fontsize = 20)
    plt.loglog(xdata, ydata, 'o', label='out'); plt.grid(True)
   # plt.subplot(3,2,s)
#    plt.grid()
#    plt.loglog(x, y, 'bo')
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.legend(['Grado'])
#    plt.xlabel('$K$', fontsize = 20)
#    plt.ylabel('$P_K$', fontsize = 20)
    
def plotDegreeDistribution_promedio(s,G):  
    #matplotlib inline
    global nx
    global degs
    global items
    global x
    global y
    global grado2
    y1 = np.zeros((G.number_of_nodes()))
    for i in range(len(G.degree(weight='weight'))): 
        y1[i]=G.degree(i,weight='weight')
    #items =  sorted(degs.items ())
#    x, y1 = np.array(degs.items()).T
#    y1= (y1*1.0)/sum(y1)
#    y1=np.sort(y1)
#    y1=np.flipud (y1)
    #y[y[:].argsort()]
#    y = [float(i) / sum(y) for i in y]
    #f, ax=plt.subplot(2,sharex=True)
   
    (f, d) = np.histogram(y1, bins=np.linspace(0,np.amax(y1+1),num=np.amax(y1)+2))
    xdata = d[1:len(d)]
    ydata = np.double(np.flipud(np.cumsum(np.flipud(f))))/sum(f) 
    grado2=ydata
  
    plt.ylabel(r'$P_K$',fontsize = 20)
    plt.loglog(xdata, ydata, 'o', label='out'); plt.grid(True)   
   
##    plt.subplot(2,3,s)
#    plt.grid()
#    plt.loglog(x, y1, 'bo') 
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.legend(['Grado'])
#    plt.xlabel('$K$', fontsize = 20)
#    plt.ylabel('$P_K$', fontsize = 20)   
#    

def existe(pila,data):
    r=0
    for p in range(len(pila)):
        if pila[p]==data:
           r=1
           break
    return r      
def disponible(conec,num):
   global aristas
   global pesos
   r=0
  
   for posicion in range(len(aristas)):
     if aristas[posicion]==[conec,num]:
         pesos[posicion]=pesos[posicion] +1
         r=1
         break
   return r  

def crear_redes_gephi(a):
   sources=list() 
   targets=list()
   tipo=list()
   label=list()
   for ind in range(len(a)):
        source,target=a[ind]
        sources.append(pila.index(source))
        targets.append(pila.index(target))
        tipo.append('Undirected')
        label.append('Uso')   
   return sources, targets,tipo,label
   
def guardar_red_gephi(sources,targets,label,tipo,pesos,nombre_nodos,nombre_red):
    global pila
    global orden_comunidad
    if len(orden_comunidad)>1:
     red_rips= {'Source':sources,'Weigth':pesos,'Target':targets,'Label':label,'Type':tipo,'Atributo_comunidad':orden_comunidad}          
    else:    
     red_rips={'Source':sources,'Weigth':pesos,'Target':targets,'Label':label,'Type':tipo}      
    
    nodos_rips={'Id':np.arange(len(pila)),'Label':pila}
    nodos_rips=pd.DataFrame(nodos_rips)
    red_rips=pd.DataFrame(red_rips)
    red_rips.to_csv(nombre_red + '.csv')
    nodos_rips.to_csv(nombre_nodos +'.csv') 
    
def crear_red(sources,targets,pesos):
    global pila
    
    G=nx.Graph()
    G.add_nodes_from([0,len(pila)-1])
    for index in range(len(pesos)):
        G.add_weighted_edges_from([(sources[index],targets[index],pesos[index])])
    return G

def grafica_entropia(x,y,c):
     plt.xlabel(r'$ \overline{w_{ij}}$',fontsize = 20)
     plt.ylabel(r'$ S $',fontsize = 20)
     plt.grid(True)
     plt.loglog(x,y,'bo',color=c)
     plt.show()

def corte_crudo():
    global pesos
    global sources
    global targets
    global p
    pe=pesos.copy()
    sou=sources.copy()
    tar=targets.copy()
 #   p=50
#    p=np.round(sum(pe)/len(pe))
#    p=p*0.1
    fin=len(pe)
    i=0
    while fin!=i:
        print(i)
        if pe[i]<=p:
           pe.pop(i)
           sou.pop(i)
           tar.pop(i)
           fin=len(pe)
        else:
          i=i+1
    return sou,tar,pe    
afiliados=pd.read_csv("2016_12_23 Consulta BDUA Risaralda.csv",sep=';',usecols=['id_ident','grupopobl','sexo','fec_nac','cond_benef','ano','mes'])  #,header=0,names=['id_ident','grupopobl','sexo','cond_benef'])
afiliados_contributivo=pd.read_csv("2017_06_28 Consulta Contr BDUA 2011 - 2015.csv",sep='|')
r=pd.read_csv("2017_04_07 cons_rips.csv",sep=';')
r.columns=['ano','tipo_doc','id_ident','fecha','consul','diag1','diag2','diag3','diag4']

#afiliados=afiliados[afiliados['grupopobl']==9]

periodo_estudio=[2011]                     
mes_seleccion=6
for periodo in periodo_estudio:
    df_afiliados=afiliados[ (afiliados['mes']==mes_seleccion)]# & (afiliados['grupopobl']==5)]
    consultas=r
    procedimiento_afiliados=consultas.merge(df_afiliados)
        
    print('Diagnosticos-Diagnostico general')
    v=list()
    c1=list()
    red_sisben=list()
    red_victima=list()
    red_todos=list()
    conexion=list()
    lista_sisben=list()
    lista_victima=list()
    lista_todos=list()
    fuentes=list()
    destinos=list()
    labels=list()
    types=list()
    pesos=list()
    pila=list() 
    aristas=list()
    conec=0
    mes_ant=0
    dia_ant=0
    num_ant=0
    pesos=list()
    pesos_cortados=list()
    pesos_cortados1=list()
    aristas_cortadas=list()
    aristas_cortadas1=list()
    np.random.seed(32)
    limite_corte=30
    limite_corte_2=20
    orden_comunidad=list()
    edad=list()
    #individuo=
    #for index in range(len(procedimiento_afiliados)):
    for u in range(100000): 
        index=np.random.randint(1,len( procedimiento_afiliados))
        
        diag1=procedimiento_afiliados['diag1'][index]  
        diag2=procedimiento_afiliados['diag2'][index]
        diag3=procedimiento_afiliados['diag3'][index]  
        diag4=procedimiento_afiliados['diag4'][index]
        s=2016-np.int(procedimiento_afiliados['fec_nac'].astype(str)[0][0:4])
        edad.append(s)
        if type(diag1)==str: 
         if type(diag2)==str: 
           if disponible(diag1,diag2)==0:
              aristas.append([diag1,diag2])
              pesos.append(1)
              if existe(pila,diag1)==0:
                  pila.append(diag1)
              if existe(pila,diag2)==0:
                 pila.append(diag2)
         if type(diag3)==str:         
            if disponible(diag1,diag3)==0:
               aristas.append([diag1,diag3])
               pesos.append(1)
               if existe(pila,diag1)==0:
                  pila.append(diag1)
               if existe(pila,diag3)==0:
                  pila.append(diag3)   
         if type(diag4)==str:        
           if disponible(diag1,diag4)==0:
              aristas.append([diag1,diag4])
              pesos.append(1)
              if existe(pila,diag1)==0:
                  pila.append(diag1)
              if existe(pila,diag4)==0:
                 pila.append(diag4)   
sources,targets,tipo,label=crear_redes_gephi(aristas)        
guardar_red_gephi(sources,targets,label,tipo,pesos,'nodos_red_completa_procedimiento_diagnostico_09_julio_2017_sisben','red_completa_procedimiento_diagnostico_09_julio_2017_sisben')
red=crear_red(sources,targets,pesos)
s=0


cli=list(nx.clique.find_cliques(red))
p=10 # la general a 15, corta bien. Sisben con 10 mejora el corte,
x1,x2,x3=corte_crudo()
red1=crear_red(x1,x2,x3)
cli2=list(nx.clique.find_cliques(red1))
k=3
adjust_cliques(k)
deteccion_overlap_cliques()
#k=4
deteccion_comunidades_overlap_cliques(k)
fil=np.where(k_comunidades==k_comunidades.max())
filtrados=pd.Series(fil[0])
filtrados=filtrados.drop_duplicates()
intensidad_clique_cortada(k)
crear_comunidades()
grafica_comunidades()

#cli2=list(nx.clique.enumerate_all_cliques(red))

#entropia_max_sin_peso= -((1.0/(red.number_of_nodes()-1))*(np.log(1.0/(red.number_of_nodes()-1))))
#K5 = nx.convert_node_labels_to_integers(red,first_label=0)
#c1 = list(nx.k_clique_communities(K5, 3))
#c2 = list(nx.k_clique_communities(K5, 4))
#list(c1[0])

#f3=a_s+(b_s*t1)
#plt.plot(t1,f3,'r')
#prome_gra_reducido=np.zeros(red.number_of_nodes())
#pro_gra,prom_wij=peso_promedio(red)
#entropia_red=np.zeros((red.number_of_nodes()))#plt.hold(True)
#plotDegreeDistribution(s,red)
#plotDegreeDistribution_promedio(s,red)
#punto_x=100
#punto1_x=400
#punto_x2=170
#punto2_x2=960
#fitting_cola()
#t=np.arange(punto_x,punto1_x)
#f1=a1[0]*np.exp(a1[1]*t)
#t1=np.arange(punto_x2,punto2_x2)
#f2=a2[0]*np.exp(a2[1]*t1)
#plt.plot(t,f1,'r')
#plt.plot(t1,f2,'r')
#plt.hold(False)
#plt.show
#(a_s,b_s,r,tt,stderr)=stats.linregress(np.arange(punto_x,punto1_x),np.log(grados_1[punto_x:punto1_x]))
#(a_s1,b_s1,r1,tt1,stderr1)=stats.linregress(np.arange(punto_x2,punto2_x2),np.log(grado2[punto_x2:punto2_x2]))
#entropia(pro_gra)
#grafica_entropia(prom_wij,entropia_red,'blue')
    
    
    ########## Proceso de corte en la red ################
    
#cortar_red()
#sources,targets,tipo,label=crear_redes_gephi(aristas_cortadas)
#
#red1=crear_red(sources,targets,pesos_cortados)
#guardar_red_gephi(sources,targets,label,tipo, pesos_cortados,'nodos_red_completa_procedimiento_diagnostico_corte_1','red_completa_procedimiento_diagnostico_corte_1')
#   
#    
#plotDegreeDistribution_promedio(s,red1)
#plotDegreeDistribution(s,red1)
#pro_gra,prom_wij=peso_promedio(red1)
#entropia_red=np.zeros((red1.number_of_nodes()))#,red.number_of_nodes()))
#entropia(pro_gra)
#grafica_entropia(prom_wij,entropia_red,'red')
#    
#cortar_red_1()
#
#sources,targets,tipo,label=crear_redes_gephi(aristas_cortadas)
#
#red2=crear_red(sources,targets,pesos_cortados)
#guardar_red_gephi(sources,targets,label,tipo, pesos_cortados,'nodos_red_completa_procedimiento_diagnostico_corte_2','red_completa_procedimiento_diagnostico_corte_2')
#   
##plotDegreeDistribution_promedio(s,red2)
##plotDegreeDistribution(s,red2)
##xs=nx.clustering(red2)
##nx.closeness.closeness_centrality(red1)
#laplaciana=nx.laplacian_spectrum(red)
#normalizado=laplaciana /sum(laplaciana)
#k=np.where(laplaciana>0)[0]
#k=np.sort(normalizado)
##
##valores=nx.eigenvector_centrality_numpy(red,weight='weight')
#entropia_valor_propio(laplaciana)
#corte_hub(red,10)
#sources,targets,tipo,label=crear_redes_gephi(aristas_cortadas1)        
#guardar_red_gephi(sources,targets,label,tipo,pesos_cortados1,'nodos_red_completa_diagnosticos_diagnostico_corte_hub','red_completa_diagnostico_diagnostico_corte_hub')
#red3=crear_red(sources,targets,pesos_cortados1)
#s=0
#plotDegreeDistribution(s,red3)
#plotDegreeDistribution_promedio(s,red3)
#
#prome_gra_reducido=np.zeros(red3.number_of_nodes())
#pro_gra,prom_wij=peso_promedio(red3)
#entropia_red=np.zeros((red3.number_of_nodes()))
#entropia(pro_gra)
#grafica_entropia(prom_wij,entropia_red,'blue')
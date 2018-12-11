# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:33:24 2018

@author: dgutierrez
"""

import numpy as np
import matplotlib.pyplot as plt

def representa_solucion(individual, datos):
    s = datos[:,0]
    z = datos[:,1]
    solucion = individual[:-5]
    
    indices = np.nonzero(solucion)[0]
    sred = s[indices]
    zred = z[indices]
    plt.plot(s, z, "b--")
    plt.plot(sred, zred, "k", marker = '*')
    plt.xlabel("Distancia [m]")
    plt.ylabel("Altura [m]")
    plt.legend(["Perfil río", "Solucion"])
    plt.annotate("Presa", (sred[-1]-80, zred[-1]+20))
    plt.annotate("Turbina", (sred[0]-80, zred[0]+20))
    plt.grid(True)
    
    
    

datos = np.loadtxt("river.csv", delimiter=";")
solucion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
representa_solucion(solucion, datos)

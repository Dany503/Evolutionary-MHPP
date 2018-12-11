# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 21:26:02 2018

@author: dgutierrez
"""

import numpy as np
from skopt import dummy_minimize
from skopt import dump
import scipy as sp
import random

def crea_individuo(size= 205):
    individuo = [0 for n in range(size)]
    x1 = random.randint(0, size-1)
    x2 = random.randint(0, size-1)
    if x1 > x2:
        x3 = x1
        x1 = x2
        x2 = x3
    elif x1 == x2:
        if x2 < size - 2:
            x2 = x2 + 2
        else:
            x1 = x1 - 2
            
    if (x2 == (x1 + 1)):
        if x2 < size -2:
            x2 = x2 + 1
        else:
            x1 = x1 - 1
       
    individuo[x1:x2] = [1 for n in range(x2-x1)]
    return individuo

def fitness_function_single(individual, datos = np.loadtxt("river.csv", delimiter=";")):
    s = datos[:,0]
    z = datos[:,1]
    solucion = individual[:-5]
    Db = individual[-5:]
    Db_str = "0b"
    for bit in Db:
        Db_str = Db_str + str(int(bit))
    D= int(Db_str, 2) * 10**-2
    if D == 0:
        D = 32e-2
   
    indices = np.nonzero(solucion)[0]
    
    # tenemos que evitar soluciones con un solo punto
    if len(indices) == 1:
        coste = 1000000
        return coste
    # tenemos que evitar que todos sean 0
    if max(solucion) == 0.0:
        coste = 1000000
        return coste  

# quitamos todos los ceros
    sred = s[indices]
    zred = z[indices]
    #print(len(sred), (zred))
    f_interpola = sp.interpolate.interp1d(sred, zred)
    indice_minimo = indices[0]
    indice_maximo = indices[-1]

    comprueba_superior = f_interpola(s[indice_minimo:indice_maximo]) - z[indice_minimo:indice_maximo]
    comprueba_inferior = -1 * comprueba_superior

    # comprobamos las restricciones del problema
    if (all(comprueba_superior <= 1.5) == False):
        coste = 1000000
        return coste
    if (all(comprueba_inferior <= 1.5) == False):
        coste = 1000000
        return coste 

    Hg = z[indice_maximo] - z[indice_minimo]
    L= np.sum(np.sqrt((sred[1:] - sred[0:-1])**2 + (zred[1:] - zred[0:-1])**2))

    RHO  = 1000
    G    = 9.8
    F    = 2e-3
    DNOZ = 22e-3
    SNOZ = (np.pi*DNOZ**2)/4
    REND = 0.9
    
    potencia = REND * (RHO/(2*SNOZ**2))*(Hg/(1/(2*G*SNOZ**2)+F*L/(D**5)))**(3/2)
    caudal = (Hg/(1/(2*G*SNOZ**2)+F*L/D**5))**(1/2)
    coste = (L + 50 * sum(solucion)) * D**2
    
    if potencia < 8e3:
        coste = 1000000
    
    if caudal > 35e-3:
        coste = 1000000

    return coste 

x0 = crea_individuo()
res = dummy_minimize(fitness_function_single, [(0, 1) for i in range(205)], x0=[x0])
fichero = open("individuos.txt", "w")
fichero2 = open("fitness.txt", "w")
for i in res.x_iters:
    fichero.write(str(i))
    fichero.write(str("\n"))

for j in res.func_vals:
    fichero2.write(str(j))
    fichero2.write("\n")
fichero.close()
fichero2.close()


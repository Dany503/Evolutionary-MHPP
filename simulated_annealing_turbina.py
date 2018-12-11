# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 12:31:15 2017

@author: dany
"""

from __future__ import print_function
import random
from simanneal import Annealer
import numpy as np
import scipy as sp

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
    
def mutacion (individuo, indpb=0.05):
    for j, i in enumerate(individuo):
        if random.random() < indpb:
            if i == 1:
                if random.random() <= 0.8:
                    individuo[j] = 0
            if i == 0:
                if random.random() <= 0.2:
                    individuo[j] = 1
    return individuo

def flipbit(individuo, indpb=0.05):
    for j, i in enumerate(individuo):
        if random.random() < indpb:
            if i == 1:
                individuo[j] = 0
            if i == 0:
                individuo[j] = 1
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

class Turbina(Annealer):
    def __init__(self, state):
        super(Turbina, self).__init__(state)  

    def move(self):
        self.state = mutacion(self.state)
        #self.state = flipbit(self.state)

    def energy(self):
        e = 0
        e = fitness_function_single(self.state)
        return e

if __name__ == '__main__':
    res_individuos = open("individuos.txt", "w")
    res_fitness = open("fitness.txt", "w")
    for i in range(5):
        init_state = crea_individuo()
        tur = Turbina(init_state)
        tur.steps = 100000
        tur.copy_strategy = "slice"
        state, e = tur.anneal()
        res_individuos.write(str(state))
        res_individuos.write("\n")
        res_fitness.write(str(e))
        res_fitness.write("\n")
        del(tur)
    res_fitness.close()
    res_individuos.close()
        
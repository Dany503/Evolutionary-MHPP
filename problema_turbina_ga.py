"""
Python: Machine Learning, Optimización y Aplicaciones
Aplicación algorimo genético al diseño de una planta hidraulica.
2018
"""

# módulos de Python que vamos a utilizar
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
    
def mutacion (individuo, indpb):
    for j, i in enumerate(individuo):
        if random.random() < indpb:
            if i == 1:
                if random.random() <= 0.8:
                    individuo[j] = 0
            if i == 0:
                if random.random() <= 0.2:
                    individuo[j] = 1
    return individuo,

def fitness_function_multiobjective(individual, datos):
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
        potencia = -1000000
        caudal = 1000000
        return potencia,caudal,coste
    # tenemos que evitar que todos sean 0
    if max(solucion) == 0.0:
        coste = 1000000
        potencia = -1000000
        caudal = 1000000
        return potencia,caudal,coste  

# quitamos todos los ceros
    sred = s[indices]
    zred = z[indices]
    f_interpola = sp.interpolate.interp1d(sred, zred)
    indice_minimo = indices[0]
    indice_maximo = indices[-1]

    comprueba_superior = f_interpola(s[indice_minimo:indice_maximo]) - z[indice_minimo:indice_maximo]
    comprueba_inferior = -1 * comprueba_superior

    # comprobamos las restricciones del problema
    if (all(comprueba_superior <= 1.5) == False):
        coste = 1000000
        potencia = -1000000
        caudal = 1000000
        return potencia,caudal,coste
    if (all(comprueba_inferior <= 1.5) == False):
        coste = 1000000
        potencia = -1000000
        caudal = 1000000
        return potencia,caudal,coste 

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
        potencia = -1000000
        caudal = 1000000
        return potencia,caudal,coste

    return potencia, caudal, coste  

def fitness_function_single(individual, datos):
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
        return coste,
    # tenemos que evitar que todos sean 0
    if max(solucion) == 0.0:
        coste = 1000000
        return coste,  

# quitamos todos los ceros
    sred = s[indices]
    zred = z[indices]
# interpolamos de forma lineal entre los puntos
    f_interpola = sp.interpolate.interp1d(sred, zred)
    indice_minimo = indices[0]
    indice_maximo = indices[-1]

    comprueba_superior = f_interpola(s[indice_minimo:indice_maximo]) - z[indice_minimo:indice_maximo]
    comprueba_inferior = -1 * comprueba_superior

    # comprobamos las restricciones del problema
    if (all(comprueba_superior <= 1.5) == False):
        coste = 1000000
        return coste,
    if (all(comprueba_inferior <= 1.5) == False):
        coste = 1000000
        return coste, 

    Hg = z[indice_maximo] - z[indice_minimo] # diferencia de altura
    L= np.sum(np.sqrt((sred[1:] - sred[0:-1])**2 + (zred[1:] - zred[0:-1])**2))

    RHO  = 1000 # densidad del agua
    G    = 9.8 # gravedad
    F    = 2e-3 # fricción kp
    DNOZ = 22e-3 # coeficiente de descarga
    SNOZ = (np.pi*DNOZ**2)/4 # sección del inyector
    REND = 0.9 # rendimiento
    
    potencia = REND * (RHO/(2*SNOZ**2))*(Hg/(1/(2*G*SNOZ**2)+F*L/(D**5)))**(3/2)
    caudal = (Hg/(1/(2*G*SNOZ**2)+F*L/D**5))**(1/2)
    coste = (L + 50 * sum(solucion)) * D**2
    
    if potencia < 8e3:
        coste = 1000000
    
    if caudal > 35e-3:
        coste = 1000000

    return coste,  

# paso1: creación del problema
#creator.create("Problema1", base.Fitness, weights=(-1,))
#descomentar para hacer el problema multiobjetivo
creator.create("Problema1", base.Fitness, weights=(1.0,-1.0,-1.0))

# paso2: creación del individuo
creator.create("Individual", list, fitness=creator.Problema1)

toolbox = base.Toolbox() # creamos la caja de herramientas
# Registramos nuevas funciones
toolbox.register("individual", tools.initIterate, creator.Individual, crea_individuo)
toolbox.register("ini_poblacion", tools.initRepeat, list, toolbox.individual)

# Operaciones genéticas
#toolbox.register("evaluate", fitness_function_single, datos = np.loadtxt("river.csv", delimiter=";"))
toolbox.register("evaluate", fitness_function_multiobjective, datos = np.loadtxt("river.csv", delimiter=";"))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutacion, indpb=0.05)
#toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# descomentar la linea siguiente para el caso multiobjetivo
toolbox.register("select", tools.selNSGA2)
#toolbox.register("select", tools.selTournament, tournsize = 3)

def unico_objetivo_ga(c, m, i):
    """ los parámetros de entrada son la probabilidad de cruce, la probabilidad 
    de mutación y el número iteración
    """
    NGEN = 100
    MU = 100 # aumentar 
    LAMBDA = 100 # aumentar
    CXPB = c
    MUTPB = m
    random.seed(i) # actualizamos la semilla cada vez que hacemos una simulación
    
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats= stats, halloffame=hof, verbose = False)
    
    return pop, hof, logbook


def multi_objetivo_ga():
    NGEN = 100
    MU = 100
    LAMBDA = 100
    CXPB = 0.6
    MUTPB = 0.4
    
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.ParetoFront()
    random.seed(64) # semilla del generador de números aleatorios
    
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              halloffame=hof)
    
    return pop, hof

# función que utilizo para visualizar la evolución
def plot(log):
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    ax1.fill_between(gen, fit_mins, fit_maxs, where=fit_maxs >= fit_mins, facecolor='g', alpha = 0.2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    ax1.set_ylim([0, 500])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    multi_objetivo = True # poner como True para que sea multi objetivo
    if multi_objetivo == True:
        res_individuos = open("individuos.txt", "w")
        res_fitness = open("fitness.txt", "w")
        pop_new, pareto_new = multi_objetivo_ga()
        for ide, ind in enumerate(pareto_new):
            res_individuos.write(str(ind))
            res_individuos.write("\n")
            res_fitness.write(str(ind.fitness.values[0]))
            res_fitness.write(",")
            res_fitness.write(str(ind.fitness.values[1]))
            res_fitness.write(",")
            res_fitness.write(str(ind.fitness.values[2]))
            res_fitness.write("\n")
        res_fitness.close()
        res_individuos.close()
    else:    
        parameters= [(0.6, 0.4)] # probabilidades que quiero probar
        for c, m in parameters:
            for i in range(0, 1): # cambiamos 1 por otro valor para ejecutar más prueba
                res_individuos = open("individuos.txt", "a")
                res_fitness = open("fitness.txt", "a")
                pop_new, pareto_new, log = unico_objetivo_ga(c, m, int(i))
                for ide, ind in enumerate(pareto_new):
                    res_individuos.write(str(i))
                    res_individuos.write(",")
                    res_individuos.write(str(ind))
                    res_individuos.write("\n")
                    res_fitness.write(str(i))
                    res_fitness.write(",")
                    res_fitness.write(str(c))
                    res_fitness.write(",")
                    res_fitness.write(str(m))
                    res_fitness.write(",")
                    res_fitness.write(str(ind.fitness.values[0]))
                    res_fitness.write("\n")
                del(pop_new)
                del(pareto_new)
                res_fitness.close()
                res_individuos.close()
        plot(log)


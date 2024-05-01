
import random
import math
import copy
from collections import defaultdict
from operator import itemgetter
import numpy as np
from Main import TSP
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import time


class EvolutionaryAlgorithm(TSP):
    def __init__(self, algorithm, filename, generations, population_size, offsprings, mutationRate, Iterations, crossoverFlag, maxminFlag):
        TSP.__init__(self, filename, generations, population_size, offsprings,
                     mutationRate, Iterations, crossoverFlag, maxminFlag)

        self.algorithm = algorithm
        self.fitness = []
        self.nodes = int

    def optimization(self):
        BSF_analysis = []
        AVG_analysis = []
        Gens = [k for k in range(self.generations)]
        if self.algorithm == "tsp":
            population, self.nodes = TSP.genpopulation(self)
            self.fitness = TSP.fitness_function(self, population)
        for i in range(self.Iterations):
            print("Iteration number: ", i)
            listt_best = []
            listt_best.append(i)
            listt_avg = []
            listt_avg.append(i)
            #listt = []
            lstnew_best = []
            lstnew_avg = []
            for k in range(self.generations):
                #gen = []
                # print("generation number", k)

                ### PARENT SELECTIONS: ###
                #parentselection = self.binarytournament(1)
                parentselection = self.FPS(1)

                children = self.crossover(parentselection)
                mutated_children = self.mutation(children)
                self.offspring_fitness(mutated_children)

                ### SURVIVOR SELECTIONS: ###
                # self.binarytournament(0)
                self.truncation()
                # self.FPS(0)

                fitness_forgraph = []
                for i in self.fitness:
                    a = np.array([i[1]])
                    countt = len(np.unique(a))
                    fitness_forgraph.append(countt)
                avg_sum = np.sum(fitness_forgraph)
                avggg = (avg_sum)/self.population_size
                lstnew_avg.append(avggg)

                b = np.array(self.fitness[0][1])
                # best_fit = len(np.unique(b))
                lstnew_best.append(len(np.unique(b)))
            listt_best.append(lstnew_best)
            listt_avg.append(lstnew_avg)
            BSF_analysis.append(listt_best)
            AVG_analysis.append(listt_avg)

        
        result = self.fitness[0][0]
        
        return result

    def BestSoFar(self, BSF_analysis):
        iterr = []
        for i in range(self.generations):
            iterr.append(0)
        for j in range(self.generations):
            summ = 0
            for k in range(self.Iterations):
                summ = summ + BSF_analysis[k][1][j]
            iterr[j] = (summ/self.Iterations)
        return iterr
        print(iterr)

    def AvgSoFar(self, AVG_analysis):
        iterr = []
        for i in range(self.generations):
            iterr.append(0)
        for j in range(self.generations):
            summ = 0
            for k in range(self.Iterations):
                summ = summ + AVG_analysis[k][1][j]
            iterr[j] = (summ/self.Iterations)
        return iterr
        print(iterr)

    def offspring_fitness(self, mutated_children):
        offspring_fitness = []
        if self.algorithm == "tsp":
            offspring_fitness = TSP.fitness_function(self, mutated_children)
        for x in offspring_fitness:
            self.fitness.append(x)

    def binarytournament(self, parentflag):
        if parentflag == 1:  # For parent selection
            parentselection = []
            parentonelist = []
            parenttwolist = []
            loopsize = int(self.offsprings/2)
            if self.maxminFlag == 1:  # Maximised fitness
                for i in range(loopsize):
                    a, b = random.sample(range(0, (self.population_size-1)), 2)
                    if self.fitness[a][0] > self.fitness[b][0]:
                        parentonelist.append(self.fitness[a][1])
                    else:
                        parentonelist.append(self.fitness[b][1])
                    c, d = random.sample(range(0, (self.population_size-1)), 2)
                    while(c == a or c == b):
                        c = random.randint(0, self.population_size - 1)
                    while(d == a or d == b or d == c):
                        d = random.randint(0, self.population_size - 1)
                    if self.fitness[c][0] > self.fitness[d][0]:
                        parenttwolist.append(self.fitness[c][1])
                    else:
                        parenttwolist.append(self.fitness[d][1])
                parentselection.append(parentonelist)
                parentselection.append(parenttwolist)
                return parentselection
            elif(self.maxminFlag == 0):  # Minimised fitness
                for i in range(loopsize):
                    a, b = random.sample(range(0, (self.population_size-1)), 2)
                    if self.fitness[a][0] < self.fitness[b][0]:
                        parentonelist.append(self.fitness[a][1])
                    else:
                        parentonelist.append(self.fitness[b][1])
                    c, d = random.sample(range(0, (self.population_size-1)), 2)
                    while(c == a or c == b):
                        c = random.randint(0, self.population_size - 1)
                    while(d == a or d == b or d == c):
                        d = random.randint(0, self.population_size - 1)
                    if self.fitness[c][0] < self.fitness[d][0]:
                        parenttwolist.append(self.fitness[c][1])
                    else:
                        parenttwolist.append(self.fitness[d][1])
                parentselection.append(parentonelist)
                parentselection.append(parenttwolist)
                return parentselection

        elif (parentflag == 0):  # Survivor Selection
            if self.maxminFlag == 1:  # Maximised fitness
                survivor = []
                while len(survivor) != self.population_size:
                    parentonelist = []
                    a, b = random.sample(range(0, (self.population_size-1)), 2)
                    if self.fitness[a][0] >= self.fitness[b][0]:
                        parentonelist.append(self.fitness[a][0])
                        parentonelist.append(self.fitness[a][1])
                    elif self.fitness[a][0] < self.fitness[b][0]:
                        parentonelist.append(self.fitness[b][0])
                        parentonelist.append(self.fitness[b][1])
                    survivor.append(parentonelist)

                self.fitness = survivor
                self.fitness = sorted(
                    self.fitness, key=itemgetter(0), reverse=True)
            elif(self.maxminFlag == 0):  # Minised fitness
                survivor = []
                while len(survivor) != self.population_size:
                    parentonelist = []
                    a, b = random.sample(range(0, (self.population_size-1)), 2)
                    if self.fitness[a][0] <= self.fitness[b][0]:
                        parentonelist.append(self.fitness[a][0])
                        parentonelist.append(self.fitness[a][1])
                    elif self.fitness[a][0] > self.fitness[b][0]:
                        parentonelist.append(self.fitness[b][0])
                        parentonelist.append(self.fitness[b][1])
                    survivor.append(parentonelist)

                self.fitness = survivor
                self.fitness = sorted(self.fitness, key=itemgetter(0))
                # print("fitness", self.fitness)

    def crossover(self, parentselection):
        children = []
        loopsize = int(self.offsprings/2)
        for k in range(loopsize):
            lst1 = parentselection[0][k]
            lst2 = parentselection[1][k]

            new_child1 = list(0 for x in range(len(lst1)))
            new_child2 = list(0 for x in range(len(lst2)))
            # to ensure breakpoint is somewhere in middle
            minn, maxx = random.sample(range(2, self.nodes-1), 2)

            if minn > maxx:
                y = maxx
                maxx = minn
                minn = y
            n = len(new_child1)

            if self.crossoverFlag == 1:  # TWO POINT CROSSOVER
                for i in range(minn, maxx+1):
                    new_child1[i] = lst1[i]

                for i in range(minn, maxx+1):
                    new_child2[i] = lst2[i]

                i = maxx+1
                count = 0
                ind = (i + count) % n

                for x in range(len(new_child1)):
                    i = (i + count) % n
                    ind = (ind + count) % n
                    if lst2[i] not in new_child1:
                        new_child1[ind] = lst2[i]
                        ind = ind+1
                    if ind == minn:
                        break
                    i = i+1
                # 2nd child
                i = maxx+1
                count = 0
                ind = (i + count) % n

                for x in range(len(new_child2)):
                    i = (i + count) % n
                    ind = (ind + count) % n
                    if lst1[i] not in new_child2:
                        new_child2[ind] = lst1[i]
                        ind = ind+1
                    if ind == minn:
                        break
                    i = i+1
            elif self.crossoverFlag == 0:  # ONE POINT CROSSOVER
                minn = random.randint(0, self.nodes - 1)
                for i in range(minn):
                    new_child1[i] = lst1[i]

                for i in range(minn):
                    new_child2[i] = lst2[i]

                for i in range(minn, n):
                    new_child1[i] = lst2[i]

                for i in range(minn, n):
                    new_child2[i] = lst1[i]
            children.append(new_child1)
            children.append(new_child2)
        return children

    def mutation(self, children):
        for i in children:
            randno = random.randint(1, 100)
            if randno < self.mutationRate*100:
                one = random.randint(0, self.nodes-1)
                two = random.randint(0, self.nodes-1)

                temp = i[one]
                i[one] = i[two]
                i[two] = temp

        return children

    def truncation(self):
        if self.maxminFlag == 0:
            self.fitness = sorted(self.fitness, key=itemgetter(0))
        elif self.maxminFlag == 1:
            self.fitness = sorted(
                self.fitness, key=itemgetter(0), reverse=True)

        self.fitness = self.fitness[:30]
        return self.fitness

    def FPS(self, parentflag):
        if parentflag == 1:  # Parent selection
            if self.maxminFlag == 0:
                parentselection = []
                parentonelist = []
                parenttwolist = []

                fitter = [0 for x in range(len(self.fitness))]
                totalfit = 0
                for i in self.fitness:
                    totalfit = totalfit+i[0]
                for j in range(len(self.fitness)):
                    fitter[j] = totalfit/(self.fitness[j][0])
                proportionsum = sum(fitter)

                for k in range(len(fitter)):
                    fitter[k] = fitter[k]/proportionsum
                cumulativeprop = [0 for x in range(len(self.fitness))]
                cumtotal = 0
                for l in range(len(fitter)):
                    cumulativeprop[l] = cumtotal+fitter[l]
                    cumtotal = cumtotal+fitter[l]
                loopsize = int(self.offsprings/2)
                for _ in range(loopsize):
                    first = random.random()
                    second = random.random()

                    for m in range(len(cumulativeprop)):
                        value = cumulativeprop[m]

                        if value >= first:
                            parentonelist.append(self.fitness[m][1])
                            break

                    for n in range(len(cumulativeprop)):
                        value1 = cumulativeprop[n]

                        if value1 >= second:
                            parenttwolist.append(self.fitness[n][1])
                            break
                parentselection.append(parentonelist)
                parentselection.append(parenttwolist)
                return parentselection
            elif(self.maxminFlag == 1):
                parentselection = []
                parentonelist = []
                parenttwolist = []

                fitter = [0 for x in range(len(self.fitness))]
                # print("fitter", fitter)
                totalfit = 0
                for i in self.fitness:
                    totalfit = totalfit+i[0]
                # print("totalfit", totalfit)
                for j in range(len(self.fitness)):
                    # print(fitness[j][0])
                    fitter[j] = self.fitness[j][0]/totalfit
                # print("updated fitter", fitter)
                proportionsum = sum(fitter)

                for k in range(len(fitter)):
                    fitter[k] = fitter[k]/proportionsum
                # print("new", fitter)
                cumulativeprop = [0 for x in range(len(self.fitness))]
                cumtotal = 0
                for l in range(len(fitter)):
                    cumulativeprop[l] = cumtotal+fitter[l]
                    cumtotal = cumtotal+fitter[l]
                loopsize = int(self.offsprings/2)
                for _ in range(loopsize):
                    first = random.random()
                    second = random.random()

                    for m in range(len(cumulativeprop)):
                        value = cumulativeprop[m]

                        if value >= first:
                            parentonelist.append(self.fitness[m][1])
                            break

                    for n in range(len(cumulativeprop)):
                        value1 = cumulativeprop[n]

                        if value1 >= second:
                            parenttwolist.append(self.fitness[n][1])
                            break
                parentselection.append(parentonelist)
                parentselection.append(parenttwolist)
                return parentselection
        elif (parentflag == 0):  # Survivor selection
            if self.maxminFlag == 1:
                parentonelist = []
                parenttwolist = []

                fitter = [0 for x in range(len(self.fitness))]
                # print("fitter", fitter)
                totalfit = 0
                for i in self.fitness:
                    totalfit = totalfit+i[0]
                # print("totalfit", totalfit)
                for j in range(len(self.fitness)):
                    # print(fitness[j][0])
                    fitter[j] = self.fitness[j][0]/totalfit
                # print("updated fitter", fitter)
                proportionsum = sum(fitter)

                for k in range(len(fitter)):
                    fitter[k] = fitter[k]/proportionsum
                # print("new", fitter)
                cumulativeprop = [0 for x in range(len(self.fitness))]
                cumtotal = 0
                for l in range(len(fitter)):
                    cumulativeprop[l] = cumtotal+fitter[l]
                    cumtotal = cumtotal+fitter[l]

                survivor = []
                while len(survivor) != self.population_size:
                    parentonelist = []
                    first = random.random()

                    for m in range(len(cumulativeprop)):
                        value = cumulativeprop[m]

                        if value >= first:
                            parentonelist.append(self.fitness[m][0])
                            parentonelist.append(self.fitness[m][1])
                            survivor.append(parentonelist)
                            break
                self.fitness = survivor
                self.fitness = sorted(
                    self.fitness, key=itemgetter(0), reverse=True)
            elif (self.maxminFlag == 0):
                parentonelist = []
                parenttwolist = []

                fitter = [0 for x in range(len(self.fitness))]
                # print("fitter", fitter)
                totalfit = 0
                for i in self.fitness:
                    totalfit = totalfit+i[0]
                # print("totalfit", totalfit)
                for j in range(len(self.fitness)):
                    # print(fitness[j][0])
                    fitter[j] = totalfit/(self.fitness[j][0])
                # print("updated fitter", fitter)
                proportionsum = sum(fitter)

                for k in range(len(fitter)):
                    fitter[k] = fitter[k]/proportionsum
                # print("new", fitter)
                cumulativeprop = [0 for x in range(len(self.fitness))]
                cumtotal = 0
                for l in range(len(fitter)):
                    cumulativeprop[l] = cumtotal+fitter[l]
                    cumtotal = cumtotal+fitter[l]

                survivor = []
                while len(survivor) != self.population_size:
                    parentonelist = []
                    first = random.random()

                    for m in range(len(cumulativeprop)):
                        value = cumulativeprop[m]

                        if value >= first:
                            parentonelist.append(self.fitness[m][0])
                            parentonelist.append(self.fitness[m][1])
                            survivor.append(parentonelist)
                            break
                self.fitness = survivor
                self.fitness = sorted(self.fitness, key=itemgetter(0))


# For cross over :
# 0 -> 1 point
# 1 -> 2 point
# For maxminFlag :
# 0 -> min
# 1 -> max

# Testing:
# (algorithm, generations, population_size, offsprings, mutationRate, Iterations, crossoverFlag, maxminFlag)

generations = 3000
pop_size = 30
offsprings = 10
mutation_rate = 0.5
iterationss = 10
start = time.time()
tsp = EvolutionaryAlgorithm("tsp", "qa194.tsp",
                            generations, pop_size, offsprings, mutation_rate, iterationss, 1, 0)

print("Optimized Solution", tsp.optimization())
end= time.time()

print("Time: ", end-start)
print("Schemes: FPS and Truncation")


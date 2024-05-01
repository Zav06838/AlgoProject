import random
import math
import copy
from collections import defaultdict
from operator import itemgetter
import numpy as np

class TSP:
    def __init__(self, file, generations, population_size, offsprings, mutationRate, Iterations, crossoverFlag, maxminFlag):
        self.generations = generations
        self.population_size = population_size
        self.offsprings = offsprings
        self.mutationRate = mutationRate
        self.Iterations = Iterations
        self.crossoverFlag = crossoverFlag
        self.maxminFlag = maxminFlag
        self.file = file

        self.clean = []
        self.distances = []

    def genpopulation(self):
        population = []
        f = open(self.file, "r")
        for line in f.readlines():
            neww = line.split()
            self.clean.append(neww)

        for i in range(7):
            del self.clean[0]
        del self.clean[-1]
        cities = len(self.clean)
        for _ in range(1, (self.population_size+1)):
            lst = []
            for i in range(1, cities+1):
                randomno = random.randint(1, cities)
                while randomno in lst:
                    randomno = random.randint(1, cities)
                lst.append(randomno)
            population.append(lst)

        self.distances = [[0 for x in range(len(self.clean))]
                          for y in range(len(self.clean))]

        for i in range(len(self.distances)):
            for j in range(len(self.distances)):
                self.distances[i][j] = round(math.sqrt(
                    ((float(self.clean[i][1])-float(self.clean[j][1]))**2)+((float(self.clean[i][2])-float(self.clean[j][2]))**2)), 4)
        print(population)
        return population, cities

    def fitness_function(self, population):
        tour = copy.deepcopy(population)
        for i in tour:
            last = i[0]
            i.append(last)

        fit = [0 for x in range(len(population))]  # list here population
        dictionary = []
        for i in range(len(population)):
            for j in range(len(population[i])):
                first = tour[i][j]
                second = tour[i][j+1]
                fit[i] += self.distances[first-1][second-1]
            dictionary.append([fit[i], population[i]])
        # print(dictionary)
        return dictionary

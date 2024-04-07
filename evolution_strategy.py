"""
WSI PROJECT 2

This program implements the evolution strategy
in order to find the minimum and maximum of the given function
"""
import numpy as np

class EvolutionStrategy():

    def __init__(self) -> None:
        self.population_size = 128
        self.offspring_size = 512
        self.iteration_num = 1000
        self.lower = -2
        self.upper = 2
        self.sigma = 0.1
        self.min_or_max = 'max'

    # returns an initial population created with random starting points
    def initial_population(self):
        return np.random.uniform(self.lower, self.upper, 2*self.population_size).reshape((self.population_size, 2))

    # defines the fitness function
    def fitness_function(self, population):
        x = population[:,0]
        y = population[:,1]
        return np.array((9*x*y)/np.exp(x**2 + 0.5*x + y**2), dtype=np.float64)
    
    # finds the best fitted element
    def find_best(self, population, fit):
        if self.min_or_max == 'min':
            best_fit = np.min(fit)
        elif self.min_or_max == 'max':
            best_fit = np.max(fit)
        
        best = population[np.where(fit == best_fit)[0][0]]
        return best, best_fit
    
    # draws random indexes and creates offspring population
    def create_offspring(self, population):
        rand_indexes = np.random.randint(0, self.population_size, self.offspring_size)
        offspring = population[rand_indexes]
        return offspring
    
    # randomly crosses parents and returns an array of children
    def crossover(self, offspring):
        children = []
        for element in offspring:
            partner = offspring[np.random.randint(0, self.offspring_size)]
            a = np.random.random()
            child = a*element + (1-a)*partner
            children.append(child)
        return np.array(children)
    
    # mutate children
    def mutate(self, children):
        noise = np.random.normal(0, self.sigma, 2*self.offspring_size).reshape((self.offspring_size, 2))
        mutants = children + noise
        return mutants
    
    # finds the function's minimum or maximum using evolution strategy
    def es_solve(self, population):
        iteration = 0
        population_fit = self.fitness_function(population)
        best, best_fit = self.find_best(population, population_fit)

        while iteration <= self.iteration_num:
            offspring = self.create_offspring(population)
            children = self.crossover(offspring)
            mutants = self.mutate(children)

            mutants_fit = self.fitness_function(mutants)
            best_mutant, best_mutant_fit = self.find_best(mutants, mutants_fit)
            if (self.min_or_max == 'min' and best_mutant_fit <= best_fit)\
                or (self.min_or_max == 'max' and best_mutant_fit >= best_fit):
                best = best_mutant
                best_fit = best_mutant_fit

            candidates = np.append(population, mutants, axis=0)
            candidates_fit = np.append(population_fit, mutants_fit, axis=0)

            sorted_indexes = np.argsort(candidates_fit)
            if self.min_or_max == 'max':
                sorted_indexes = np.flip(sorted_indexes)

            candidates = candidates[sorted_indexes]
            candidates_fit = candidates_fit[sorted_indexes]

            population = candidates[:self.population_size]
            population_fit = candidates_fit[:self.population_size]

            iteration += 1
        return best, best_fit
    

    

if __name__ == "__main__":
    solution = EvolutionStrategy()
    population = solution.initial_population()
    # population = 10*np.ones((solution.population_size,2))
    print()
    print(solution.es_solve(population))
    print()
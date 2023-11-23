import logging 
import numpy as np
from logging.handlers import RotatingFileHandler

def format_bounds(bounds):
    bounds_str = "["
    for row in bounds:
        bounds_str += "[" + " ".join(map(str, row)) + "] "
    return bounds_str.strip() + "]"

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('Normal_logger')
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler('logging_de.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

error_logger = logging.getLogger('Error_logger')
error_logger.setLevel(logging.ERROR)
error_file_handler = RotatingFileHandler('errors.log', mode='w')
error_file_handler.setFormatter(formatter)
error_logger.addHandler(error_file_handler)

class DifferentialEvolution:
    def __init__(self, fobj, bounds, mutation_coefficient=0.8, crossover_coefficient=0.7, population_size=20):

        self.fobj = fobj
        self.bounds = bounds
        self.mutation_coefficient = mutation_coefficient
        self.crossover_coefficient = crossover_coefficient
        self.population_size = population_size
        self.dimensions = len(self.bounds)

        self.a = None
        self.b = None
        self.c = None
        self.mutant = None
        self.population = None
        self.idxs = None
        self.fitness = []
        self.min_bound = None
        self.max_bound = None
        self.diff = None
        self.population_denorm = None
        self.best_idx = None
        self.best = None
        self.cross_points = None

    def _init_population(self):
        self.population = np.random.rand(self.population_size, self.dimensions)
        self.min_bound, self.max_bound = self.bounds.T
        
        self.diff = np.fabs(self.min_bound - self.max_bound)
        self.population_denorm = self.min_bound + self.population * self.diff
        self.fitness = np.asarray([self.fobj(ind) for ind in self.population_denorm])

        self.best_idx = np.argmin(self.fitness)
        self.best = self.population_denorm[self.best_idx]
    
    def _mutation(self):
        self.a, self.b, self.c = self.population[np.random.choice(self.idxs, 3, replace = False)]
        self.mutant = np.clip(self.a + self.mutation_coefficient * (self.b - self.c), 0, 1)
        return self.mutant
    
    def _crossover(self):
        cross_points = np.random.rand(self.dimensions) < self.crossover_coefficient
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        return cross_points

    def _recombination(self, population_index):

        trial = np.where(self.cross_points, self.mutant, self.population[population_index])
        trial_denorm = self.min_bound + trial * self.diff
        return trial, trial_denorm
    
    def _evaluate(self, result_of_evolution, population_index):
        if result_of_evolution < self.fitness[population_index]:
                self.fitness[population_index] = result_of_evolution
                self.population[population_index] = self.trial
                if result_of_evolution < self.fitness[self.best_idx]:
                    self.best_idx = population_index
                    self.best = self.trial_denorm
                if result_of_evolution > 1e-3:
                    formatted_bounds = format_bounds(self.bounds)
                    error_logger.error(f"Результат: {result_of_evolution}, превышает 1e-3. \nИспользуемые параметры: размер популяции {self.population_size}, границы {formatted_bounds}, коэффициент мутации {self.mutation_coefficient}, коэффициент скрещивания {self.crossover_coefficient}")
                    if result_of_evolution > 1e-1:
                        error_logger.critical(f"Результат: {result_of_evolution}, превышает 1e-1. \nИспользуемые параметры: размер популяции {self.population_size}, границы {formatted_bounds}, коэффициент мутации {self.mutation_coefficient}, коэффициент скрещивания {self.crossover_coefficient}")

    def iterate(self):
    
        for population_index in range(self.population_size):
            self.idxs = [idx for idx in range(self.population_size) if idx != population_index]

            self.mutant = self._mutation()
            self.cross_points = self._crossover()

            self.trial, self.trial_denorm = self._recombination(population_index)
    
            result_of_evolution = self.fobj(self.trial_denorm)

            self._evaluate(result_of_evolution, population_index)


def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))


if __name__ == "__main__":

    function_obj = rastrigin
    bounds_array = np.array([[[-20, 20], [-20, 20]], [[-10, 50], [-10, 60]], [[0, 110], [-42, 32]]])
    steps_array = [40, 100, 200]
    mutation_coefficient_array = [0.5, 0.6, 0.3]
    crossover_coefficient_array = [0.5, 0.6, 0.3]
    population_size_array = [20, 30, 40, 50, 60]

    for bounds in bounds_array:
        for steps in steps_array:
            for mutation_coefficient in mutation_coefficient_array:
                for crossover_coefficient in crossover_coefficient_array:
                    for population_size in population_size_array:

                        de_solver = DifferentialEvolution(function_obj, bounds, mutation_coefficient=mutation_coefficient, crossover_coefficient=crossover_coefficient, population_size=population_size)
                
                        de_solver._init_population()
                        formatted_bounds = format_bounds(bounds)
                        logger.info(
                            f"Инициализируем популяцию с размером: {population_size}, Границы: {formatted_bounds}, \n"
                            f"Коэффициент мутации: {mutation_coefficient}, \n"
                            f"Коэффициент скрещивания: {crossover_coefficient}, Количество итераций: {steps}, \n"
                            f"Начальная популяция: {de_solver.population_denorm}"
                        )
                        for _ in range(steps):
                            de_solver.iterate()
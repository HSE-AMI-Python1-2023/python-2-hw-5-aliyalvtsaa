import pytest 
import numpy as np

from differential_evolution import DifferentialEvolution

# CONSTANTS

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))

BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin


"""
Ваша задача добиться 100% покрытия тестами DifferentialEvolution
Различные этапы тестирования логики разделяйте на различные функции
Запуск команды тестирования:
pytest -s test_de.py --cov-report=json --cov
"""

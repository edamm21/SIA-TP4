import random
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import statistics

class Oja:

    def __init__(self, names, data, learning_rate):
        self.names = names
        self.data = self.pre_process(data)
        self.learning_rate = learning_rate
        self.input_neurons = len(data[0])
        self.w = [0 for w in range(self.input_neurons)]
        for i in range(len(self.w)):
            self.w[i] = random.uniform(-0.5, 0.5)
        self.epochs = 10000

    def pre_process(self, grid):
        normalized_columns = []
        for i in range(len(grid[0])):
            entire_col = [row[i] for row in grid]
            media = sum(entire_col) / len(entire_col)
            std = statistics.stdev(entire_col)
            normalized_column = [(value - media) / std for value in entire_col]
            normalized_columns.append(normalized_column)
        array = np.array([np.array(xi) for xi in normalized_columns])
        array = array.transpose()
        return list(array)

    def calculate(self, input):
        y = 0.0
        for j in range(len(input)):
            y += self.w[j] * input[j]
        return y

    def train(self):
        data = self.data
        t = 1
        for n in range(self.epochs):
            random.shuffle(data)
            for x in data:
                y = self.calculate(x)
                N = len(x)
                for j in range(N):
                    self.w[j] += (self.learning_rate/t) * (y * x[j] - y * y * self.w[j])
                t += 1

    def test(self, names, data):
        data = self.pre_process(data)
        for name, x in zip(names, data):
            print(name, self.calculate(x))

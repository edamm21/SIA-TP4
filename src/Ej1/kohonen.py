import random
import math
class Kohonen:

    def __init__(self, names, data, learning_rate, neighbors):
        self.data = data
        self.learning_rate = learning_rate
        self.input_neurons = len(data[0]) - 1 # porque el primer parametro es el nombre
        self.epochs = 500 * self.input_neurons
        input_dimension = len(data[0])
        self.output_grid = self.initialize_output_grid(input_dimension)
        self.neighbors = neighbors
    
    def initialize_output_grid(self, size):
        grid = [[[0 for x in range(size)] for y in range(5)] for z in range(5)] # elegí 5 por poner un numero
        for i in range(5):
            for j in range(5):
                for k in range(size):
                    grid[i][j][k] = random.uniform(-1.0, 1.0)
        return grid

    def train(self):
        indexes = list(range(0, len(self.data)))
        for epoch in range(self.epochs):
            random.shuffle(indexes)
            for idx in indexes:
                alpha = self.learning_rate # actualizar
                proximity = self.neighbors # actualizar
                bmu = self.get_bmu(self.data[idx])
                close_neurons = self.get_close_neurons(bmu, proximity)
                self.update_weights(bmu, alpha)
    
    def get_close_neurons(self, bmu, proximity):
        neighbors_to_drag = []
        for i in range(len(self.output_grid)):
            for j in range(len(self.output_grid[0])):
                curr_neuron = [i, j]
                if self.close_enough(bmu, curr_neuron, proximity):
                    neighbors_to_drag.append(curr_neuron)
        return neighbors_to_drag

    def close_enough(self, this, other, proximity):
        return self.calculate_distance(this, other) <= proximity

    def calculate_distance(self, _input, neuron):
        acum = 0.0
        for i in range(len(_input.length)):
            acum += (_input[i] - neuron[i])**2
        return math.sqrt(acum)

    def get_bmu(self, _input):
        min_distance = 1e10
        bmu = (0, 0)
        for i in range(len(self.output_grid)):
            for j in range(len(self.output_grid[0])):
                curr_distance = self.calculate_distance(_input, self.output_grid[i][j])
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    bmu = (i, j)
        return bmu

    #def initialize_output_grid(self):

    #def initialize_output_weights(self):

import random
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import statistics

class Kohonen:

    def __init__(self, names, data, learning_rate, neighbors):
        self.names = names
        self.data = self.pre_process(data)
        self.learning_rate = learning_rate
        self.input_neurons = len(data[0])
        self.epochs = 500 * self.input_neurons 
        self.neighbors = neighbors
        self.output_grid = self.initialize_output_grid(self.input_neurons, self.neighbors)
    
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
         
    def initialize_output_grid(self, size, neighborhood_size):
        grid = [[[0 for x in range(size)] for y in range(neighborhood_size)] for z in range(neighborhood_size)]
        for i in range(neighborhood_size):
            for j in range(neighborhood_size):
                for k in range(size):
                    grid[i][j][k] = random.uniform(-1.0, 1.0)
        return grid
    
    def get_close_neurons(self, bmu, proximity):
        neighbors_to_drag = []
        for i in range(len(self.output_grid)):
            for j in range(len(self.output_grid[0])):
                curr_neuron = [i, j]
                if self.close_enough(bmu, curr_neuron, proximity):
                    neighbors_to_drag.append(curr_neuron)
        return neighbors_to_drag

    def close_enough(self, this, other, proximity):
        neighborhood_influence = math.exp(-(self.calculate_distance(this, other)**2)/(2 * proximity**2))
        return neighborhood_influence < proximity

    # euclidean
    def calculate_distance(self, _input, neuron):
        acum = 0.0
        for i in range(len(_input)):
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

    def update_weights(self, N_k_t, n_t, X_p): # mantengo lo del ppt, son Nk(t), n(t) y Xp de la diapo 19
        for neighbor in N_k_t:
            row  = neighbor[0]
            col  = neighbor[1]
            diff = [n_t * (x_p_i - w_i) for x_p_i, w_i in zip(X_p, self.output_grid[row][col])]
            add  = [w_i + diff_i for w_i, diff_i in zip(self.output_grid[row][col], diff)]
            self.output_grid[row][col] = add
    
    # ref: https://towardsdatascience.com/kohonen-self-organizing-maps-a29040d688da
    def train(self):
        indexes = list(range(0, len(self.data)))
        t = 1
        _lambda = 100 # constante de tiempo
        for epoch in range(self.epochs):
            random.shuffle(indexes)
            for idx in indexes:
                n_t = self.learning_rate * math.exp(-t/_lambda) # le puse el nombre como en el ppt n(t)
                R_t = self.neighbors * math.exp(-t/_lambda) + 1 # le puse el nombre como en el ppt R(t) ==> entiendo que dice que tiene que converger a 1 al final por eso el +1
                bmu = self.get_bmu(self.data[idx])
                close_neurons = self.get_close_neurons(bmu, R_t)
                self.update_weights(close_neurons, n_t, self.data[idx])
            t += 1

    def test(self):
        x_min = y_min = -1
        x_max = y_max = len(self.output_grid)
        all_points = [None] * (len(self.output_grid)**2)
        point = 0
        for i in range(len(self.output_grid)):
            for j in range(len(self.output_grid)):
                all_points[point] = (i, j)
                point += 1
        bmus = [None] * len(self.data)
        qtys = [0] * (len(self.output_grid)**2)
        names_by_neuron = {}
        for i in range(len(self.data)):
            bmus[i] = self.get_bmu(self.data[i])
            if str(bmus[i]) in names_by_neuron:
                names_by_neuron[str(bmus[i])].append(self.names[i])
            else:
                names_by_neuron[str(bmus[i])] = [self.names[i]] 
            idx_qty = bmus[i][0] * len(self.output_grid) + bmus[i][1]
            qtys[idx_qty] += 1
            print('{}: neuron {}'.format(self.names[i], bmus[i]))
        print(names_by_neuron)
        qtys = [i * 50 for i in qtys if i > 0]
        activated_neurons = list(set(bmus));
        activated_neurons.sort()
        fig, ax = plt.subplots()
        ax.scatter(*zip(*all_points), label='Neuronas no activadas')
        ax.scatter(*zip(*activated_neurons), s=qtys, label='Neuronas activadas y su cantidad')
        ax.set_title('Activación de neuronas')
        ax.set_xlabel('Coordenada x - Neuronas')
        ax.set_ylabel('Coordenada y - Neuronas')
        for i in range(len(activated_neurons)):
            ax.annotate(int(qtys[i] / 50), (activated_neurons[i]))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend(fontsize = 8, loc = 'upper left', markerscale = 0.5)
        plt.show()
        self.create_u_matrix()

    def create_u_matrix(self):
        u_array = [0.0] * (len(self.output_grid)**2)
        all_points = [None] * (len(self.output_grid)**2)
        point = 0
        for i in range(len(self.output_grid)):
            for j in range(len(self.output_grid)):
                all_points[point] = (i, j)
                point += 1
        for i in range(len(self.output_grid)):
            for j in range(len(self.output_grid)):
                current_neuron_weights = self.output_grid[i][j]
                distance_total = 0.0
                added = 0
                if i-1 >= 0: #N
                    added += 1
                    distance_total += self.calculate_distance(current_neuron_weights, self.output_grid[i-1][j])
                if i+1 < len(self.output_grid): #S
                    added += 1
                    distance_total += self.calculate_distance(current_neuron_weights, self.output_grid[i+1][j])
                if j-1 >= 0: #W
                    added += 1
                    distance_total += self.calculate_distance(current_neuron_weights, self.output_grid[i][j-1])
                if j+1 < len(self.output_grid): #E
                    added += 1
                    distance_total += self.calculate_distance(current_neuron_weights, self.output_grid[i][j+1])
                if i-1 >= 0 and j-1 >= 0: #NW
                    added += 1
                    distance_total += self.calculate_distance(current_neuron_weights, self.output_grid[i-1][j-1])
                if i-1 >= 0 and j+1 < len(self.output_grid): #NE
                    added += 1
                    distance_total += self.calculate_distance(current_neuron_weights, self.output_grid[i-1][j+1])
                if i+1 < len(self.output_grid) and j-1 >= 0: #SW
                    added += 1
                    distance_total += self.calculate_distance(current_neuron_weights, self.output_grid[i+1][j-1])
                if i+1 < len(self.output_grid) and j+1 < len(self.output_grid): #SE
                    added += 1
                    distance_total += self.calculate_distance(current_neuron_weights, self.output_grid[i+1][j+1])
                distance = distance_total / added
                u_array[i * len(self.output_grid) + j] += distance
        normalize_colors = colors.Normalize(vmin=min(u_array), vmax=max(u_array))
        plt.scatter(*zip(*all_points), c=u_array, s=50);
        plt.colorbar()
        plt.gray()
        plt.title('Matríz U')
        plt.show()

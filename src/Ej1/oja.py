import random
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import statistics

class Oja:

    def __init__(self, names, data, learning_rate):
        self.names = names
        self.data = self.pre_process(data)
        self.learning_rate = learning_rate
        self.input_neurons = len(data[0])
        self.w = [0 for w in range(self.input_neurons)]
        for i in range(len(self.w)):
            self.w[i] = random.uniform(-1, 1)
        self.epochs = 5000

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

    def pca(self, path):
        data = pd.read_csv(path)
        features = ['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
        X_standarized = data[features]
        X_standarized = pd.DataFrame(data=self.pre_process(X_standarized.to_numpy()))
        correlations_matrix = X_standarized.corr()
        eigen_values_s, eigen_vectors_s = np.linalg.eig(correlations_matrix)
        idx_order_s = eigen_values_s.argsort()[::-1]
        eigen_values_s = eigen_values_s[idx_order_s]
        eigen_vectors_s = eigen_vectors_s[:,idx_order_s]
        pca_s = PCA()
        main_components_s = pca_s.fit_transform(X_standarized)
        loadings_s = pca_s.components_.T
        loadings_matrix_s = pca_s.components_.T * np.sqrt(pca_s.explained_variance_)
        names = list(data['Country'])
        y1_s = pca_s.fit_transform(X_standarized)[:,0]
        return loadings_s.transpose()[0], y1_s

    def train(self, path):
        np.set_printoptions(suppress=True,linewidth=np.nan)
        real_w, real_scores = self.pca(path)
        data = self.data
        t = 1
        weight_error = np.zeros((self.epochs, len(self.w)))
        weight_error_inv = np.zeros((self.epochs, len(self.w)))
        for n in range(self.epochs):
            random.shuffle(data)
            for x in data:
                y = self.calculate(x)
                N = len(x)
                for j in range(N):
                    self.w[j] += (self.learning_rate/t) * (y * x[j] - y * y * self.w[j])
                t += 1
            weight_error[n] = abs(self.w - real_w)
            weight_error_inv[n] = abs(self.w + real_w)

        # Si los signos quedaron dados vuelta, invertirlos
        if self.w[0] * real_w[0] < 0:
            self.w = [-self.w[i] for i in range(len(self.w))]
            print("Weights signs had to be inverted to match expected order")
            weight_error = weight_error_inv
        w_names = ['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
        for i in range(len(weight_error[0])):
            plt.plot(range(self.epochs), weight_error[:,i], label="W{} - {}".format(i+1, w_names[i]))
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title('Weight error')
        plt.show()


        print("Final weight error:")
        print(weight_error[-1])

    def test(self, names, data, path):
        real_w, real_scores = self.pca(path)
        data = self.pre_process(data)
        i = 0
        print("\n{0:<10}\t{1:<20}\t{2:<20}\t{3:<20}".format("COUNTRY","SCORE","EXPECTED","ERROR"))
        for name, x in zip(names, data):
            score = self.calculate(x)
            real_score = real_scores[i]
            error = abs(score - real_score)
            print("{0:<10}\t{1:<10}\t{2:<10}\t{3:<10}".format(name, score, real_score, error))
            i += 1

# Lo dejo por las dudas
import numpy as np
import time

class Hopfield:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.initializeW()

    def initializeW(self):
        self.W = np.zeros((len(self.alphabet[0]), len(self.alphabet[0])))
        for i in range(len(self.W)):
            for j in range(len(self.W)):
                if i != j:
                    for k in range(len(self.alphabet)): # cantidad de patrones (o letras conocidas)
                        self.W[i][j] += self.alphabet[k][i] * self.alphabet[k][j]
                    self.W[i][j] *= (1/(len(self.alphabet[0])))
    
    def sign(self, vector):
        out = np.zeros(np.shape(vector))
        for i in range(len(vector)):
            if vector[i] == 0.0:
                out[i] = 0.0
            elif vector[i] > 0.0:
                out[i] = 1.0
            else:
                out[i] = -1.0
        return out

    def not_a_known_pattern(self, vector, other):
        boolean = [False, False, False, False, False]
        for i in range(len(self.alphabet)):
            boolean[i] = np.array_equal(self.alphabet[i], vector)
        boolean[-1] = np.array_equal(other, vector)
        return not any(boolean)

    def print_nice(self, vector):
        vector = np.asarray(vector).reshape(5, 5)
        nice = ['', '', '', '', '']
        for i in range(0, 5):
            for j in range(0, 5):
                if vector[i][j] == 1: nice[i] += ('*')
                else: nice[i] += ' '
            print(nice[i])
        print('----------------')


    def algorithm(self, pattern):
        if np.shape(pattern) != np.shape(self.alphabet[0]):
            print('Mal patrón: dado es de forma {}, necesito {}'.format(np.shape(pattern), np.shape(self.alphabet[0])))
            exit()
        S = np.asarray(pattern)
        prevS = np.zeros(np.shape(pattern))
        iterations = 0
        while(self.not_a_known_pattern(S, prevS)):
            prevS = S
            S = self.sign(self.W.dot(prevS))
            self.print_nice(prevS)
            iterations += 1
        print('End pattern')
        self.print_nice(S)
        print('iterations: {}'.format(iterations))
        return S
        
        
        
        
        
    

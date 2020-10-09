import csv
import sys
import operator
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CSV_PATH = '../Ej1/data/europe.csv'

def is_number(elem):
    try:
        float(elem)
        return True
    except ValueError:
        return False

def parseCSV():
    data = open(CSV_PATH, newline='')
    reader = csv.reader(data, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONE)
    parsed_data = [[] for x in range(28)]
    parsed_country_names = ['' for x in range(28)]
    idx = 0
    for row in reader:
        if idx > 0:
            for elem in row:
                if is_number(elem):
                    parsed_data[idx - 1].append(float(elem))
                else:
                    parsed_country_names[idx - 1] = elem.replace('"', '')

        idx += 1
    return parsed_country_names, parsed_data

# PASO 1: Tomar dataset X
np.set_printoptions(linewidth=np.inf)
names, data = parseCSV()
x = data
y = names

# PASO 2: Obtener medias
medians = np.zeros(len(data[0]))
for country in data:
    medians += country
for var in range(len(medians)):
    medians[var] = medians[var] / len(data)
for i in range(len(data)):
    data[i] = data[i] - medians
#print(data)

# PASO 3: Matriz de covarianza
m = len(data)       # m = paises
n = len(medians)    # n = variables
S = np.zeros((n, n))
for i in range(n):
    for k in range(n):
        sum = 0
        for j in range(m-1):
            sum += (data[j][i] - medians[i]) * (data[j][k] - medians[k])
        S[i][k] = (1.0/n) + sum
#print(S)

# PASO 4: Calcular autovalores y autovectores de la matriz de covarianza y ordenar
lambdas, eigenvectors = np.linalg.eig(S)
idx = lambdas.argsort()[::-1]
lambdas = lambdas[idx]
eigenvectors = eigenvectors[:,idx]

# PASO 5: Formar la matriz E tomando los autovectores correspondientes a los mayores autovalores
E = eigenvectors

# PASO 6: Calcular Y = E*(X-X')
Y = np.zeros((m,n))
for country in range(m):
    for index in range(n):
        Y[country][index] = 0
        for row in range(n):
            Y[country][index] += E[row][index] * data[country][row]

# PASO 7: Ordenar
values = []
for country in range(m):
    values.append(Y[country][0])
zippy = sorted(zip(names, values), key=operator.itemgetter(1))




####################### USANDO LIBRERÍA #####################################

sc = StandardScaler()
x = sc.fit_transform(x)
pca = PCA(n_components=1)
x = pca.fit_transform(data)
explained_variance = pca.explained_variance_ratio_
print("El componente principal Y1 contiene el {}{} de la informacion".format(explained_variance[0]*100, '%'))
ranking = zip(names, x)
ranking = sorted(ranking, key = lambda x: x[1])
for country, score in ranking:
    print(country, score[0])

#Si las variables originales no estan correlacionadas,
#entonces no tiene sentido realizar un analisis de componentes principales

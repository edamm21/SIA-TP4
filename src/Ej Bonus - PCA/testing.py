import matplotlib.pyplot as plt
import pandas as pd
import statistics
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px


def standarize(grid):
    standarized_columns = []
    for i in range(len(grid[0])):    
        entire_col = [row[i] for row in grid]
        media = sum(entire_col) / len(entire_col)
        std = statistics.stdev(entire_col)
        standarized_column = [(value - media) / std for value in entire_col]
        standarized_columns.append(standarized_column)
    array = np.array([np.array(xi) for xi in standarized_columns])
    array = array.transpose()
    return array

data = pd.read_csv('files/europe.csv')
features = ['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
X_standarized = data[features]
X_standarized = pd.DataFrame(data=standarize(X_standarized.to_numpy()))
pca_s = PCA()
main_components_s = pca_s.fit_transform(X_standarized)
loadings_s = pca_s.components_.T
print('Cargas: ', loadings_s)
loadings_matrix_s = pca_s.components_.T * np.sqrt(pca_s.explained_variance_)
fig = px.scatter(main_components_s, x=0, y=1, color=data["Country"])
for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings_matrix_s[i, 0],
        y1=loadings_matrix_s[i, 1]
    )
    fig.add_annotation(
        x=loadings_matrix_s[i, 0],
        y=loadings_matrix_s[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
fig.show()
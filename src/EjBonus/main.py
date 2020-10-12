import plotly.express as px
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

CSV_PATH = 'files/europe.csv'

my_csv = CSV_PATH
df = pd.read_csv(my_csv)
features = ['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
X = df[features]

# Normalizar
X[['Area']] -= min(X[['Area'][0]])
X[['Area']] /= max(X[['Area'][0]])
X[['GDP']] -= min(X[['GDP'][0]])
X[['GDP']] /= max(X[['GDP'][0]])
X[['Inflation']] -= min(X[['Inflation'][0]])
X[['Inflation']] /= max(X[['Inflation'][0]])
X[['Life.expect']] -= min(X[['Life.expect'][0]])
X[['Life.expect']] /= max(X[['Life.expect'][0]])
X[['Military']] -= min(X[['Military'][0]])
X[['Military']] /= max(X[['Military'][0]])
X[['Pop.growth']] -= min(X[['Pop.growth'][0]])
X[['Pop.growth']] /= max(X[['Pop.growth'][0]])
X[['Unemployment']] -= min(X[['Unemployment'][0]])
X[['Unemployment']] /= max(X[['Unemployment'][0]])
# End of normalization

pca = PCA()
components = pca.fit_transform(X)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig = px.scatter(components, x=0, y=1, color=df["Country"])

# Graph vectors
for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
fig.show()

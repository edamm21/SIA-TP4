import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'files/europe_positive.csv'

my_csv = CSV_PATH
df = pd.read_csv(my_csv)
to_boxplot = pd.DataFrame(data=df)
to_boxplot.plot(kind='box')
plt.show()
features = ['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
X = df[features]
pca = PCA()
components = pca.fit_transform(X)

print('Primer componente principal: {}'.format(components[0]))

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

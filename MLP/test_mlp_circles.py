import numpy as np
from ploting import MLP_binary_draw
from MLP import MLP
import pandas as pd

df = pd.read_csv('circles.csv')

X = df.iloc[:, :2].values.T
Y = df.iloc[:, 2:].values.T

net = MLP((2, 50, 1), output_activation = 'logistic')
net.fit(X, Y)

MLP_binary_draw(X, Y, net, "circles_image.png")
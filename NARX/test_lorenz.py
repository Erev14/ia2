import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NARX import *
import pandas as pd
import statsmodels.api as sm

def lorenz(u,t):
    """
    Sistema de Lorenz

    """
    s=10; r=28; b=2.667
    ux_dot = s*(u[1] - u[0])
    uy_dot = r*u[0] - u[1] - u[0]*u[2]
    uz_dot = u[0]*u[1] - b*u[2]
    return [ux_dot, uy_dot, uz_dot]

#Condicion inicial
u0 = [0., 1., 1.05]

#Tiempo
t = np.linspace(0,100, 10000)

#Soluciones
u = odeint(lorenz, u0, t)

narx = NARX(3, 3, 10,
            dense_hidden_layers=(100,),
            learning_rate=0.01, n_repeat_train=1)

y_est = np.zeros((3,10000))
u = u.T

for i in range(10000-1):
    x = u[:,i].reshape(-1,1)
    y = u[:,i+1].reshape(-1,1)
    y_est[:,i] = narx.predict_and_train(x, y).ravel()

# Grafica
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(u[0,:], u[1,:], u[2,:], lw=0.7)
ax.plot(y_est[0,:], y_est[1,:], y_est[2,:], lw=0.5)
ax.set_xlabel("Eje x")
ax.set_ylabel("Eje y")
ax.set_zlabel("Eje z")
ax.set_title("Atractor de Lorenz")
plt.show()
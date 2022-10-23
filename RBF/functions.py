import numpy as np
def gausian(r, sigma):
    np.exp(np.power(-r, 2) / np.power(sigma, 2))

def multi_cadratic(r, epsilon):
    np.sqrt(1 + np.power(epsilon * r, 2))

def multi_cuadratic_inverse(r, epsilon):
   1 / np.sqrt(1 + np.power(epsilon * r, 2))

def spline_poliarmonic(r, k):
    odd = r[r % 2 != 0]
    even = r[r % 2 == 0]
    np.power(odd, k)
    np.power(even, k)
    even * np.log(k)
    np.power(even, k)

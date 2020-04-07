import numpy as np

def random_uniform(a, b):
    return (b - a)*np.random.random_sample() + a

def sine_function(amp, phi, x):
    return amp*np.sin(x + phi)
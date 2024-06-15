import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre, factorial
from matplotlib.colors import LinearSegmentedColormap

def R_func(r, l, n, a):  #Parte Radial
    L = genlaguerre(n - l - 1, 2 * l + 1)(r / a)
    return np.exp(-r / (2 * a)) * r**l * L

def Y_func(theta, phi, l, m):   #Parte Angular
    return sph_harm(m, l, phi, theta).real

def normalization_constant(n, l, a):  #Constante normalización
    return np.sqrt((1 / (n * a)**3) * (factorial(n - l - 1) / (2 * n * factorial(n + l))))

def u(r, theta, phi, l, m, n, a):   #\psi(r,\theta,\phi_{0})
    N = normalization_constant(n, l, a)
    return N * R_func(r, l, n, a) * Y_func(theta, phi, l, m)


# Parámetros del átomo de hidrógeno
n = 1  # Parámetro n para los polinomios de Laguerre /
l = 0  # Grado del armónico esférico /
m = 0  # Orden del armónico esférico / momento magnético

a = 5.29177e-5  # Radio de Bohr (aprox)

r_max = 5 * a  # Máximo radio a considerar

phi_fixed = np.pi / 2  


Nr = 100
Ntheta = 100
r = np.linspace(0.1 * a, r_max, Nr)  
theta = np.linspace(0, 2 * np.pi, Ntheta)
R, Theta = np.meshgrid(r, theta, indexing='ij')

U = u(R, Theta, phi_fixed, l, m, n, a)

#Graficar: 

cmap = LinearSegmentedColormap.from_list('red_to_black', ['black','orange'])

plt.style.use('dark_background')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)
levels = 300
c = ax.contourf(Theta, R / a, U, levels, cmap=cmap)  # Normalizar el radio por el radio de Bohr

cbar = fig.colorbar(c, ax=ax)
cbar.set_ticks([]) 

ax.set_title(rf'$({n}, \, {l}, \, {m})$', fontsize=20, color='white')
ax.set_yticklabels([])  
ax.set_xticklabels([])  

ax.tick_params(colors='white', which='both')  

plt.show()

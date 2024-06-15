import numpy as np
import matplotlib.pyplot as plt

# Parámetros
m = 4*910.0  # Masa de la partícula
hbar = 6.626  # Constante de Planck reducida
E = 0.5  # Energía
U = 1.0  # Potencial
a = 0.2  # Ancho del potencial

# Calcular b1 y b2
b1 = np.sqrt(2 * m * E / hbar**2)
b2 = np.sqrt(2 * m * (U - E) / hbar**2)

# Calcular la constante A3 (la tomamos como real)
A3 = 1.0

# Calcular los coeficientes A1, B1, A2, B2, y B3
A1 = 0.5 * A3 * np.exp(1j * b1 * a) * (2 * np.cosh(b2 * a) + 1j * ((b2 / b1) - (b1 / b2)) * np.sinh(b2 * a))
B1 = -0.5j * A3 * np.exp(1j * b1 * a) * ((b2 / b1) + (b1 / b2)) * np.sinh(b2 * a)
A2 = 0.5 * (1 + 1j * b1 / b2) * A3 * np.exp(1j * b1 * a - b2 * a)
B2 = 0.5 * (1 - 1j * b1 / b2) * A3 * np.exp(1j * b1 * a + b2 * a)
B3 = 0.0

# Definir las funciones de onda para las tres regiones
def psi_1(x):
    return A1 * np.exp(1j * b1 * x) + B1 * np.exp(-1j * b1 * x)

def psi_2(x):
    return A2 * np.exp(b2 * x) + B2 * np.exp(-b2 * x)

def psi_3(x):
    return A3 * np.exp(1j * b1 * (x))

# Rango de x
x1 = np.linspace(-1, 0, 400)
x2 = np.linspace(0, a, 400)
x3 = np.linspace(a, a + 1, 400)

# Graficar las funciones de onda
plt.figure(figsize=(12, 6))

plt.plot(x1, np.real(psi_1(x1)), label=r'Real($\psi(x)$)', linestyle='-',color='blue')
plt.plot(x1, np.imag(psi_1(x1)), label=r'Imag($\psi(x)$)', linestyle='-',color='red')
plt.plot(x1, np.abs(psi_1(x1)), label=r'$|\psi(x)|$', linestyle='-', color='purple')

plt.plot(x2, np.real(psi_2(x2)), linestyle='--',color='blue')
plt.plot(x2, np.imag(psi_2(x2)), linestyle='--',color='red')
plt.plot(x2, np.abs(psi_2(x2)), linestyle='-', color='purple')

plt.plot(x3, np.real(psi_3(x3)), linestyle='-',color='blue')
plt.plot(x3, np.imag(psi_3(x3)), linestyle='-',color='red')
plt.plot(x3, np.abs(psi_3(x3)), linestyle='-', color='purple')

plt.fill_between([0, a], -6, 6, color='gray', alpha=0.3, label='Barrera de potencial')

plt.xlabel('x')
plt.ylabel(r'$\psi(x)$')
plt.title('Función de onda $\psi(x)$ bajo efecto túnel')
plt.axvline(x=0, color='gray', linestyle='--')
plt.axvline(x=a, color='gray', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()

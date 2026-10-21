import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---- Objetivo: Red Force (Ferrari Land) ----
g = 9.81
v_objetivo = 180 / 3.6             # velocidad en lo más bajo: 50 m/s
t_caida    = 6.0                   # tiempo de caída del borde al fondo [s]
alto = v_objetivo**2 / (2 * g)     # altura para 180 km/h [m] ≈ 127.4

# ---- Cálculo de 'a' para que la caída dure t_caida ----
# Con s = 4 a h (= pendiente en el borde al cuadrado):
# t = sqrt(2h/(g s)) * integral_0^{pi/2} sqrt(1 + s sin^2(th)) dth
def tiempo_caida(s):
    n = 20000
    th = (np.arange(n) + 0.5) * (np.pi / 2) / n          # regla del punto medio
    I = np.sum(np.sqrt(1 + s * np.sin(th)**2)) * (np.pi / 2) / n
    return np.sqrt(2 * alto / (g * s)) * I

t_min = np.sqrt(2 * alto / g)      # caída vertical (límite): no se puede bajar de aquí
if t_caida <= t_min:
    raise ValueError(f"Con esta altura el tiempo mínimo es {t_min:.2f} s")

lo, hi = 1e-3, 1e4                 # bisección (el tiempo decrece al aumentar s)
for _ in range(100):
    s = np.sqrt(lo * hi)
    if tiempo_caida(s) > t_caida:
        lo = s
    else:
        hi = s

a     = s / (4 * alto)             # z = a x^2
largo = 2 * np.sqrt(alto / a)      # anchura total [m]

# ---- Condiciones iniciales ----
x0 = largo / 2                     # cae desde el punto más alto (borde)
v0 = 0.0

# ---- Tiempo ----
tf  = 24                           # tiempo simulado [s] (aprox. un periodo)
fps = 25                           # fotogramas por segundo del GIF
velocidad = 2                      # 1 = tiempo real, 2 = el doble de rápido...
N   = int(tf * fps / velocidad)    # menos fotogramas -> GIF más rápido
salto = 1

# ---- Opciones del GIF ----
guardar_gif = True
nombre_gif  = "red_force.gif"
try:
    carpeta = os.path.dirname(os.path.abspath(__file__))
except NameError:
    carpeta = os.getcwd()
archivo_gif = os.path.join(carpeta, nombre_gif)

# ---- Ecuación del movimiento: y = [x, v] ----
k = 2 * g * a

def f(y):
    x, v = y
    acc = -(4 * a**2 * x * v**2 + k * x) / (1 + 4 * a**2 * x**2)
    return np.array([v, acc])

# ---- Runge-Kutta 4 (paso fijo) ----
def rk4_paso(y, h):
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

t  = np.linspace(0, tf, N)
dt = t[1] - t[0]
sub = 20
h   = dt / sub

y = np.zeros((N, 2))
y[0] = [x0, v0]
for i in range(1, N):
    yi = y[i - 1].copy()
    for _ in range(sub):
        yi = rk4_paso(yi, h)
    y[i] = yi

x = y[:, 0]
z = a * x**2
v = np.sqrt(y[:, 1]**2 * (1 + 4 * a**2 * x**2))     # módulo de la velocidad [m/s]

i_fondo = np.argmax(x <= 0)                          # primer fotograma en el fondo
print(f"Altura: {alto:.1f} m | Anchura: {largo:.1f} m | a = {a:.4e} 1/m")
print(f"Pendiente en el borde: {np.degrees(np.arctan(2*a*x0)):.0f} grados")
print(f"Tiempo de caída: {t[i_fondo]:.2f} s (objetivo {t_caida} s)")
print(f"Velocidad máxima: {v.max():.1f} m/s = {v.max()*3.6:.0f} km/h")
print(f"Duración del GIF: {N / fps:.1f} s (velocidad x{velocidad})")

# ---- Figura ----
fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
ax.set_facecolor("white")

xr = np.linspace(-largo / 2, largo / 2, 400)
a_txt = f"{a:.4f}".replace(".", ",")
ax.plot(xr, a * xr**2, "k", linewidth=2.5,
        label=rf"$z = ax^2$,  $a = {a_txt}\ \mathrm{{m^{{-1}}}}$")
ax.set_aspect("equal")
ax.set_xlim(-largo / 2 - 15, largo / 2 + 15)
ax.set_ylim(-10, alto + 30)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_title("Modelo de Atracción Fija")
ax.grid(True)

estela, = ax.plot([], [], "b", linewidth=1)
carro,  = ax.plot([x[0]], [z[0]], "ro", markersize=9)
texto = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", fontsize=11)
ax.legend(loc="upper center", fontsize=11)

# ---- Animación ----
def actualizar(i):
    carro.set_data([x[i]], [z[i]])
    estela.set_data(x[:i + 1], z[:i + 1])
    texto.set_text(f"t = {t[i]:.1f} s\nv = {v[i]*3.6:.0f} km/h")
    return carro, estela, texto

frames = range(0, N, salto)
anim = FuncAnimation(fig, actualizar, frames=frames,
                     interval=1000 / fps, blit=True)

if guardar_gif:
    anim.save(archivo_gif, writer=PillowWriter(fps=fps))
    print(f"GIF guardado en: {archivo_gif}")

plt.show()
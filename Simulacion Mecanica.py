import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =====================================================
#  ELECCIÓN DE SIMULACIÓN: ATRACCIÓN FIJA (B), MÓVIL (C) 
# =====================================================
print("Selecciona la simulación:")
print("1) Atracción fija (apartado b)")
print("2) Atracción móvil (apartado c)")
opcion = input("\nOpción (1,2): ").strip()

if opcion == "1":
    simulacion = "B"
elif opcion == "2":
    simulacion = "C"
else:
    print("Opción incorrecta. Se selecciona la fija por defecto")
    simulacion = "B"
# =====================================================
#  PARÁMETROS (SE PUEDEN CAMBIAR AQUÍ)
# =====================================================
a  = 0.5        # z = a x^2  // (o r^2 si es C) >> El parámetro a también se puede cambiar como condición inicial.
g  = 9.81

# Condiciones iniciales 
x0 = 2        # Posición Inicial [m]  (en C es la distancia inicial al eje, r0)
v0 = 0.0        # Velocidad Inicial [m/s] (en C es la velocidad radial, rdot0)
w  = 2       # Velocidad Angular [rad/s]  

# Tiempo y Fotogramas 
tf = 5       # tiempo total [s] (puede convenir cambiarlo según los parámetros)
N  = 250        # nº de fotogramas
salto = 2       # 2 para ir formando el gif más rápido

# Opciones del GIF  
guardar_gif = True # Cuidado: destruye el gif anterior y lo sustituye por el nuevo con otro nombre.
nombre_gif  = "atraccionfija.gif" if simulacion.upper() == "B" else "atraccion_movil.gif"  

# Código para llevar el gif a la misma carpeta que el script en el guardado. 
try:
    carpeta = os.path.dirname(os.path.abspath(__file__))
except NameError:        
    carpeta = os.getcwd()
archivo_gif = os.path.join(carpeta, nombre_gif)

# =====================================================
#  ECUACIÓN DEL MOVIMIENTO: y = [q, qdot]  Como tenemos la misma en B y C, la definimos una vez y cambiamos solo q y k según toque.
#  B: q = x,  k = 2ga  
#  C: q = r,  k = 2ga - w^2
# =====================================================
simulacion = simulacion.upper()
if simulacion == "B":
    k = 2 * g * a
elif simulacion == "C":
    k = 2 * g * a - w**2
else:
    raise ValueError('La simulacion debe ser "B" o "C"')

def f(y):
    q, v = y   #La derivada de la posición es la velocidad
    acc = -(4 * a**2 * q * v**2 + k * q) / (1 + 4 * a**2 * q**2)  #La aceleración despeja de la ec. de movimiento, al definir k ya se utiliza correcto.
    return np.array([v, acc]) 

# ---- Integración con Runge-Kutta 4 (paso fijo) ----
def rk4_paso(y, h):
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

t  = np.linspace(0, tf, N)
dt = t[1] - t[0]
sub = 20                      # subpasos de RK4 entre fotogramas
h   = dt / sub

y = np.zeros((N, 2))
y[0] = [x0, v0]
for i in range(1, N):
    yi = y[i - 1].copy()
    for _ in range(sub):
        yi = rk4_paso(yi, h)
    y[i] = yi

q = y[:, 0]                   # x (B) o r (C)
R = 1.2 * np.max(np.abs(q))   # Tamaño de la ventana
rr = np.linspace(-R, R, 200)  # Puntos del raíl

# =====================================================
#  FIGURA Y ANIMACIÓN
# =====================================================
if simulacion == "B":
    # Apartado B: plano xz 
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white")  
    ax.set_facecolor("white")

    ax.plot(rr, a * rr**2, "k", linewidth=2, label="z = ax$^2$") #Ploteamos la parábola que representa el raíl.
    ax.set_aspect("equal") #Para que esté equilibrada la parábola y no se vea deformada. 
    ax.set_xlim(-R, R)
    ax.set_ylim(0, a * R**2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Modelo de Atracción Fija") 
    ax.grid(True)

    estela, = ax.plot([], [], "b", linewidth=1) #Representa la trayectoria que va siguiendo el carro
    carro,  = ax.plot([q[0]], [a * q[0]**2], "ro", markersize=10) #Representa la masa puntual
    ax.legend(loc="upper center", fontsize=12)

    def actualizar(i):
        carro.set_data([q[i]], [a * q[i]**2])
        estela.set_data(q[:i + 1], a * q[:i + 1]**2) 
        return carro, estela

    usar_blit = True #Redibuja solo los objetos que necesita ir actualizando y no toda la figura.

else:
    # Apartado C: 3D, raíl giratorio 
    phi = w * t                       # ligadura phi = w t
    X = q * np.cos(phi)               # Tal y como calculamos las equivalencias en el anexo
    Y = q * np.sin(phi)
    Z = a * q**2

    fig = plt.figure(figsize=(7, 5.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(0, a * R**2)
    ax.set_box_aspect((2 * R, 2 * R, a * R**2))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Modelo de Atracción Móvil")

    rail,   = ax.plot(rr * np.cos(phi[0]), rr * np.sin(phi[0]), a * rr**2,
                      "k", linewidth=2, label="z = ar$^2$")
    estela, = ax.plot([], [], [], "b", linewidth=1)
    carro,  = ax.plot([X[0]], [Y[0]], [Z[0]], "ro", markersize=10)
    ax.legend(loc="upper right", fontsize=12)

    def actualizar(i):
        rail.set_data_3d(rr * np.cos(phi[i]), rr * np.sin(phi[i]), a * rr**2)
        carro.set_data_3d([X[i]], [Y[i]], [Z[i]])
        estela.set_data_3d(X[:i + 1], Y[:i + 1], Z[:i + 1])
        return rail, carro, estela

    usar_blit = False                 # el blitting no funciona en 3D

frames = range(0, N, salto)
anim = FuncAnimation(fig, actualizar, frames=frames,
                     interval=1000 * dt * salto, blit=usar_blit)

# ---- Guardar el GIF ----
if guardar_gif:
    anim.save(archivo_gif, writer=PillowWriter(fps=round(1 / (dt * salto))))
    print(f"GIF guardado en: {archivo_gif}")

plt.show() 
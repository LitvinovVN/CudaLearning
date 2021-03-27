from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def z_function(x, y):
    return 0.026 - 0.0000002 * y - 0.00016*np.log(x) - 0.00077*np.log(y)

x = np.linspace(1.6, 640, 100)
y = np.linspace(1000, 30000, 50)

X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)

# построение поверхности,
# с указанием шага сетки на поверхности (rstride, cstride),
# цветовой карты (cmap) и цвета границы на поверхности (edgecolor)
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='jet', edgecolor='k')
#ax.contourf(X, Y, Z, 100, cmap='jet')

# установка белого фона сетки
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# подписи (заголовки) осей,
# с указанием размера и цвета шрифта
# текст в $$ - текст в формате LaTex
ax.set_xlabel('$k$', fontsize = 15, color = 'black')
ax.set_ylabel('$Y$', fontsize = 15, color = 'black')
ax.set_zlabel('$T_{GPU}, ms$', fontsize = 15, color = 'black')

#ax.contour(Z)

# создание цветовой шкалы (colorbar)
m = cm.ScalarMappable(cmap = 'jet') 
m.set_array(Z) 
plt.colorbar(m,shrink = 0.1) 

#plt.savefig('Stat_3D.svg', fmt='svg')

plt.show()

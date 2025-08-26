# FISCA_ESTADISTICA
Tareas de Simulación para Física Estadística

Para hacer que este código funcione se debe tener instalado numpy y matplotlib de lo contrario usar "pip install numpy matplotlib" . Luego es necesario que corra el módulo de view.py e indicar en la terminal los parámetros requeridos.  

Dentro del módulo view.py. 

La función main simplemente es un menú para ejecutar los puntos 7 y 8 de la tarea. 

Al elegir el "1" en el menú, tendremos el histograma con una gaussiana teórica que debería tener como resultado para el punto 7 y un mensaje sobre la comparación con el teorema del límite central.  Se reecomienda el uso de N = 1000 y simulaciones = 10000

Al elegir el "2" en el menú, tendremos un scaterring con una linealización teórica que debería tener como resultado para el punto 8 y un mensaje sobre el coeficiente de difusión.
Se recomienda el uso del rango N=100..5000 steps de 100

Supuestos: pasos ±1 (equivale a= 1), tiempo discreto Δt=1; por tanto, teórico  𝐷 =1/2.

Por otro lado, en el modulo logic tendremos muchas funciones. 

Tenemos la primera sección que corresponde al punto 7 cada función tienen una descripción en la documentanción. Sin embargo, el sistema básico de la primera sección es que voy a crear múltiples marchas aleatorias y de forma simultánea voy a tomar los datos teórios según el teorema central del límite. Con el fin de poder crear un histograma basado en la simulación aleatoria que con el paso del tiempo debería parecerse a la distribución normal que se verá también en la gráfica. Además de un mensaje de comparación teórica. 

En la segunda sección que corresponde al punto 8 cada función tiene una descripción en la documentación. Sin embargo, el sistema básico de la segunda sección es que voy a tomar n simulaciones de múltiples marchas aleatorias y con esa información sacar la ecuación de difusión acorde a la relación  lineal ⟨x2⟩≈2DNΔt. Con el fin de crear un scaterring de los puntos aleatorios con una linealización teórica de ⟨x2⟩≈2DNΔt. Además de un mensaje de que usamos esa linealización debido a que se cumple el teorema central como se ve en el punto 7 y una conclusión que espero sea suficiente. 




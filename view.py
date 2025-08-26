import numpy as np
import matplotlib.pyplot as plt
from logic import run_simulation, run_multiple_n_simulations, calculate_diffusion_constant, plot_histogram, plot_time_series_sample, plot_mean_squared_vs_n


def main():
    print("=== Simulación de Marcha Aleatoria Unidimensional ===")
    
    print("1. Simulación individual con histograma")
    print("2. Análisis de ⟨x²⟩ vs N para múltiples valores")
    
    option = input("\nSeleccione opción (1 o 2): ")
    
    if option == "1": #PUNTO 7
        # Simulación individual con histograma
        
        n_steps = int(input("Ingrese el número de pasos por marcha aleatoria (N): "))
        n_simulations = int(input("Ingrese el número de simulaciones a ejecutar: "))
        results = run_simulation(n_steps, n_simulations)
            
        plot_histogram(results)
        print(f'Al realizar muchas simulaciones de la marcha aleatoria con un número grande de pasos  N, \
              el histograma de las posiciones finales se aproxima notablemente a la forma de una campana gaussiana. \
              Este resultado concuerda con el teorema central del límite, que establece que la suma de un gran número \
              de variables aleatorias independientes y equiprobables tiende a seguir una distribución normal,\
              independientemente de la distribución original de cada paso (en este caso, saltos discretos de ±a con igual probabilidad).\
              Así, aunque cada trayectoria individual es errática, el comportamiento colectivo de la partícula a gran escala queda\
              descrito por una distribución normal de media cero y varianza proporcional al número de pasos.')
        
    elif option == "2": #PUNTO 8
        # Análisis de ⟨x²⟩ vs N
        
        n_simulations = int(input("Ingrese el número de simulaciones para cada N: "))
        n_min = int(input("Valor mínimo de N: "))
        n_max = int(input("Valor máximo de N: "))
        n_step = int(input("Incremento de N: "))


        n_values = list(range(n_min, n_max + 1, n_step))
        

        results_by_n = run_multiple_n_simulations(n_values, n_simulations)
        diffusion_constant, n_values_list, mean_squared_list, coefficients = calculate_diffusion_constant(results_by_n)
              
        # Mostrar tabla de resultados
        print(f"\nTabla de resultados:")
        print(f"{'N':>6} {'⟨x²⟩ simulado':>15} {'⟨x²⟩ teórico':>15} {'Diferencia':>12} {'Error %':>10}")
        print("-" * 70)
        for n in n_values:
            actual = results_by_n[n]['mean_squared_position']
            theoretical = n
            diff = abs(actual - theoretical)
            error_pct = diff / theoretical * 100
            print(f"{n:6d} {actual:15.4f} {theoretical:15.4f} {diff:12.4f} {error_pct:10.2f}%")
        
        # Graficar ⟨x²⟩ vs N
        plot_mean_squared_vs_n(results_by_n, diffusion_constant, coefficients)
        
        print(f'Una vez vemos que se cumple el teorema del limite central para esta simulación. Si se calcula el crecimiento de  ⟨x2⟩ \
            con el número de pasos, se obtiene una relación lineal ⟨x2⟩≈2DNΔt,  de la cual se deduce la constante de difusión  \
            D. De esta forma, el experimento computacional no solo confirma el teorema central del límite, sino que también \
            conecta directamente la caminata aleatoria discreta con la ecuación de difusión en el límite continuo.')
        
    else:
        print("Opción no válida. Saliendo.")

if __name__ == "__main__":
    main()
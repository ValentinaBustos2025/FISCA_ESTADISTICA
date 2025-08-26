import numpy as np
import matplotlib.pyplot as plt
from logic import run_simulation, run_multiple_n_simulations, calculate_diffusion_constant, plot_histogram


    
def ejectuar_plot_histogram(results):
        plot_histogram(results)

def plot_time_series_sample(results: dict, sample_size: int = 50):
    """
    Muestra una muestra de las trayectorias temporales.
    
    Args:
        results: Diccionario con resultados
        sample_size: Número de trayectorias a mostrar
    """
    n_steps = results['n_steps']
    
    plt.figure(figsize=(12, 8))
    
    # Mostrar solo una muestra de trayectorias
    for i in range(min(sample_size, 50)):
        # Simular una trayectoria completa
        steps = np.random.choice([-1, 1], size=n_steps)
        trajectory = np.cumsum(steps)
        
        plt.plot(range(n_steps), trajectory, alpha=0.3, linewidth=0.5)
    
    plt.xlabel('Número de pasos', fontsize=12)
    plt.ylabel('Posición', fontsize=12)
    plt.title(f'Muestra de {min(sample_size, 50)} trayectorias de la marcha aleatoria\n'
              f'(N = {n_steps} pasos)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_mean_squared_vs_n(results_by_n: dict[int, dict], diffusion_constant: float, coefficients: list):
    """
    Grafica ⟨x²⟩ vs N y muestra el ajuste lineal.
    
    Args:
        results_by_n: Diccionario con resultados para diferentes N
        diffusion_constant: Constante de difusión calculada
        coefficients: Coeficientes del ajuste lineal
    """
    n_values = list(results_by_n.keys())
    mean_squared_values = [results_by_n[n]['mean_squared_position'] for n in n_values]
    theoretical_values = n_values  # ⟨x²⟩ teórico = N
    
    plt.figure(figsize=(12, 8))
    
    # Datos simulados
    plt.scatter(n_values, mean_squared_values, color='blue', s=50, alpha=0.7, label='Datos simulados')
    
    # Ajuste lineal
    fit_line = np.poly1d(coefficients)
    x_fit = np.linspace(min(n_values), max(n_values), 100)
    y_fit = fit_line(x_fit)
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Ajuste lineal: ⟨x²⟩ = {coefficients[0]:.3f}N + {coefficients[1]:.3f}')
    
    # Línea teórica
    plt.plot(n_values, theoretical_values, 'g--', linewidth=2, label='Teórico: ⟨x²⟩ = N')
    
    plt.xlabel('Número de pasos (N)', fontsize=12)
    plt.ylabel('⟨x²⟩ (Posición cuadrática media)', fontsize=12)
    plt.title('Relación ⟨x²⟩ vs N en Marcha Aleatoria Unidimensional\n'
              f'Constante de difusión D = {diffusion_constant:.4f}', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Añadir información del ajuste
    textstr = '\n'.join((
        f'Pendiente del ajuste: {coefficients[0]:.4f}',
        f'Constante de difusión D: {diffusion_constant:.4f}',
        f'Teórico: D = 0.5',
        f'Desviación: {abs(diffusion_constant - 0.5)/0.5*100:.2f}%',
        f'Ecuación: ⟨x²⟩ = 2D × N',
        f'Número de puntos: {len(n_values)}'))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

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
  
        
    elif option == "2":
        # Análisis de ⟨x²⟩ vs N
        n_simulations = int(input("Ingrese el número de simulaciones para cada N: "))
        
        # Definir valores de N a probar
        n_min = int(input("Valor mínimo de N: "))
        n_max = int(input("Valor máximo de N: "))
        n_step = int(input("Incremento de N: "))
        
        n_values = list(range(n_min, n_max + 1, n_step))
        
        print(f"\nAnalizando {len(n_values)} valores de N desde {n_min} hasta {n_max}...")
        
        # Ejecutar simulaciones para múltiples valores de N
        results_by_n = run_multiple_n_simulations(n_values, n_simulations)
        
        # Calcular constante de difusión
        diffusion_constant, n_values_list, mean_squared_list, coefficients = calculate_diffusion_constant(results_by_n)
        
        # Mostrar resultados
        print(f"\n=== Resultados del análisis ⟨x²⟩ vs N ===")
        print(f"Número de valores de N analizados: {len(n_values)}")
        print(f"Número de simulaciones por N: {n_simulations}")
        print(f"Pendiente del ajuste lineal: {coefficients[0]:.6f}")
        print(f"Constante de difusión D: {diffusion_constant:.6f}")
        print(f"Valor teórico de D: 0.5")
        print(f"Error relativo: {abs(diffusion_constant - 0.5)/0.5*100:.4f}%")
        
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
        
    else:
        print("Opción no válida. Saliendo.")

if __name__ == "__main__":
    main()
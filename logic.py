import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

#PUNTO 7

#Simulación de una marcha aleatoria unidimensional
def single_random_walk(n_steps: int, a: float = 1.0) -> float:
    steps = np.random.choice([-a, a], size=n_steps)
    return np.sum(steps)

def multiple_random_walks(n_steps: int, n_simulations: int) -> List[int]:
    """
    Ejecuta múltiples simulaciones de marchas aleatorias.
    """
    final_positions = []
    for _ in range(n_simulations):
        final_pos = single_random_walk(n_steps)
        final_positions.append(final_pos)
    return final_positions

#Cálculo de parámetros teóricos y distribución gaussiana según el teorema del límite central
def calculate_gaussian_parameters(n_steps: int) -> Tuple[float, float]:
    """
    Calcula los parámetros de la distribución gaussiana teórica.
    """
    mu = 0 ; sigma = np.sqrt(n_steps)  # TODO: Para una marcha aleatoria: μ = 0, σ = √N ** NO sé donde dice eso en el libro jeje 
    return mu, sigma

def gaussian_distribution(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Función de densidad de probabilidad gaussiana.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def run_simulation(n_steps: int, n_simulations: int) -> dict:
    """
    Ejecuta la simulación completa y devuelve los resultados.
    """
    #Simulación
    final_positions = multiple_random_walks(n_steps, n_simulations)
    mean_position = np.mean(final_positions);std_position = np.std(final_positions)
    
    
    # Parámetros teóricos y calculo de la distribución gaussiana
    mu_theoretical, sigma_theoretical = calculate_gaussian_parameters(n_steps)
    x_range = np.linspace(min(final_positions), max(final_positions), 1000)
    gaussian_theoretical = gaussian_distribution(x_range, mu_theoretical, sigma_theoretical)
    
    # Preparar datos para el histograma
    hist_values, bin_edges = np.histogram(final_positions, bins='auto', density=True) #Histograma de las posiciones finales
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 #Centros de los bins para graficar
    
    final_positions = np.array(final_positions, dtype=float)
    mean_squared_position = np.mean(final_positions**2)

    results = {
        'final_positions': final_positions,
        'mean_position': mean_position,
        'std_position': std_position,
        'mu_theoretical': mu_theoretical,
        'sigma_theoretical': sigma_theoretical,
        'hist_values': hist_values,
        'bin_centers': bin_centers,
        'x_range': x_range,
        'gaussian_theoretical': gaussian_theoretical,
        'n_steps': n_steps,
        'n_simulations': n_simulations,
        'mean_squared_position': mean_squared_position
    }
    
    return results

def plot_histogram(results: dict):
    """
    Crea y muestra el histograma de las posiciones finales.
    """
    plt.figure(figsize=(12, 8))
    
    plt.hist(results['final_positions'], 
             bins='auto', 
             density=True, 
             alpha=0.7, 
             color='lightblue',
             edgecolor='black',
             linewidth=0.5,
             label=f'Simulación ({results["n_simulations"]} realizaciones)')
    
    #Teórico
    plt.plot(results['x_range'], 
             results['gaussian_theoretical'], 
             'r-', 
             linewidth=2, 
             label='Distribución teórica (Gaussiana)')
    
    # Configuración del gráfico
    plt.xlabel('Posición final (x)', fontsize=12)
    plt.ylabel('Densidad de probabilidad', fontsize=12)
    plt.title(f'Marcha Aleatoria Unidimensional\n'
              f'N = {results["n_steps"]} pasos, '
              f'{results["n_simulations"]} simulaciones\n'
              f'Media = {results["mean_position"]:.2f}, '
              f'σ = {results["std_position"]:.2f} '
              f'(teórico: √N = {results["sigma_theoretical"]:.2f})', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Añadir texto con información estadística
    textstr = '\n'.join((
        f'Media simulada: {results["mean_position"]:.2f}',
        f'Desviación simulada: {results["std_position"]:.2f}',
        f'Desviación teórica: {results["sigma_theoretical"]:.2f}',
        f'Número de pasos: {results["n_steps"]}',
        f'Número de simulaciones: {results["n_simulations"]}'))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

#PUNTO 8

def run_multiple_n_simulations(n_values: List[int], n_simulations: int) -> dict[int, dict]: 
    """
    Ejecuta simulaciones para múltiples valores de N.
    """
    results_by_n = {}
    
    for i, n_steps in enumerate(n_values):
        results = run_simulation(n_steps, n_simulations)
        results_by_n[n_steps] = results
    
    return results_by_n

def calculate_diffusion_constant(results_by_n: dict[int, dict], delta_t: float = 1.0):
    """
    Estima D del ajuste lineal de <x^2> vs N usando <x^2> ≈ 2 D N Δt.
    """
    n_values = sorted(results_by_n.keys())
    mean_squared_values = [float(results_by_n[n]['mean_squared_position']) for n in n_values]
    coeffs = np.polyfit(np.array(n_values, float), np.array(mean_squared_values, float), 1)
    slope = float(coeffs[0])              # d<x^2>/dN
    D = slope / (2.0 * delta_t)
    return D, n_values, mean_squared_values, coeffs


def plot_time_series_sample(results: dict, sample_size: int = 50):
    """
    Muestra una muestra de las trayectorias temporales.
    """
    n_steps = results['n_steps']
    
    plt.figure(figsize=(12, 8))
    
    for i in range(min(sample_size, 50)):
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
    """
    n_values = list(results_by_n.keys())
    mean_squared_values = [results_by_n[n]['mean_squared_position'] for n in n_values]
    theoretical_values = n_values  # ⟨x²⟩ teórico = N
    
    plt.figure(figsize=(12, 8))
    plt.scatter(n_values, mean_squared_values, color='blue', s=50, alpha=0.7, label='Datos simulados')
    
    # Ajuste lineal
    fit_line = np.poly1d(coefficients)
    x_fit = np.linspace(min(n_values), max(n_values), 100)
    y_fit = fit_line(x_fit)
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Ajuste lineal: ⟨x²⟩ = {coefficients[0]:.3f}N + {coefficients[1]:.3f}')
    
    # Línea teórica
    plt.plot(n_values, theoretical_values, 'g--', linewidth=2, label='Teórico: ⟨x²⟩ = N')

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
    
    
    plt.xlabel('Número de pasos (N)', fontsize=12)
    plt.ylabel('⟨x²⟩ (Posición cuadrática media)', fontsize=12)
    plt.title('Relación ⟨x²⟩ vs N en Marcha Aleatoria Unidimensional\n'
              f'Constante de difusión D = {diffusion_constant:.4f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
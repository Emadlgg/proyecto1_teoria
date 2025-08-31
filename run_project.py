#!/usr/bin/env python3
"""
Script principal para ejecutar el proyecto de Teoría de la Computación
"""

import sys
import os

# Añadir el directorio src al path para importar los módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.automata_project import *

def main():
    """Función principal del programa"""
    print("=== Proyecto Teoría de la Computación ===")
    print("Implementación de Algoritmos para Autómatas Finitos\n")
    
    # Crear directorios de salida si no existen
    os.makedirs("output/json", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)
    
    # Solicitar expresión regular
    regex = input("Ingrese la expresión regular: ")
    print(f"\nProcesando expresión regular: {regex}")
    
    try:
        # 1. Convertir a postfix (Shunting Yard)
        processor = RegexProcessor()
        postfix = processor.shunting_yard(regex)
        print(f"Notación postfix: {postfix}")
        
        # 2. Convertir regex a AFN (Thompson)
        thompson = ThompsonNFA()
        nfa = thompson.regex_to_nfa(regex)
        print(f"AFN creado con {len(nfa.states)} estados")
        
        # Guardar y visualizar AFN
        nfa.save_to_file("output/json/nfa.json")
        print("AFN guardado en 'output/json/nfa.json'")
        
        # Mostrar AFN en formato texto
        nfa.visualize("output/visualizations/nfa_graph", "AFN (Thompson)", show_ascii=True)
        
        # 3. Convertir AFN a AFD (Construcción de Subconjuntos)
        subset_constructor = SubsetConstruction()
        dfa = subset_constructor.nfa_to_dfa(nfa)
        print(f"AFD creado con {len(dfa.states)} estados")
        
        # Guardar AFD
        dfa.save_to_file("output/json/dfa.json")
        print("AFD guardado en 'output/json/dfa.json'")
        
        # Mostrar AFD en formato texto
        dfa.visualize("output/visualizations/dfa_graph", "AFD (Subconjuntos)", show_ascii=True)
        
        # 4. Minimizar AFD (Hopcroft)
        minimizer = HopcroftMinimization()
        min_dfa = minimizer.minimize_dfa(dfa)
        print(f"AFD minimizado creado con {len(min_dfa.states)} estados")
        
        # Guardar AFD minimizado
        min_dfa.save_to_file("output/json/min_dfa.json")
        print("AFD minimizado guardado en 'output/json/min_dfa.json'")
        
        # Mostrar AFD minimizado en formato texto
        min_dfa.visualize("output/visualizations/min_dfa_graph", "AFD Minimizado (Hopcroft)", show_ascii=True)
        
        # 5. Simulaciones
        simulator = DFASimulator()
        
        print("\n" + "="*50)
        print("SIMULACIONES")
        print("="*50)
        
        while True:
            test_string = input("\nIngrese una cadena para probar (o 'quit' para salir): ")
            if test_string.lower() == 'quit':
                break
            
            accepted, trace = simulator.simulate(min_dfa, test_string)
            simulator.print_simulation(trace, accepted, test_string)
    
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        print("Verifica que la expresión regular sea válida.")

def run_examples():
    """Ejecuta los ejemplos de prueba"""
    print("Ejecutando ejemplos de prueba...")
    from src.example_usage import test_examples, detailed_simulation_example, step_by_step_example
    
    test_examples()
    detailed_simulation_example() 
    step_by_step_example()

if __name__ == "__main__":
    print("¿Qué desea ejecutar?")
    print("1. Programa principal")
    print("2. Ejemplos de prueba")
    
    choice = input("Seleccione una opción (1 o 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        run_examples()
    else:
        print("Opción inválida. Ejecutando programa principal...")
        main()
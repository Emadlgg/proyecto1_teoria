"""
Ejemplos de uso del proyecto de Teoría de la Computación
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from automata_project import *

def test_examples():
    """Ejecuta varios ejemplos de prueba"""
    
    examples = [
        "(a|b)*abb(a|b)*",
        "a*b+", 
        "(a|b)*",
        "ab*c",
        "a(b|c)*d"
    ]
    
    test_strings = [
        ["aabb", "babb", "ababb", "abbab", "abb"],
        ["ab", "abb", "abbb", "aab", "aaab"],
        ["", "a", "b", "ab", "ba", "aabb"],
        ["ac", "abc", "abbc", "abbbc", "aac"],
        ["ad", "abd", "acd", "abcd", "acbd"]
    ]
    
    # Crear directorios de salida
    os.makedirs("output/json", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)
    
    for i, regex in enumerate(examples):
        print(f"\n{'='*60}")
        print(f"EJEMPLO {i+1}: {regex}")
        print(f"{'='*60}")
        
        try:
            # Procesar expresión regular
            processor = RegexProcessor()
            postfix = processor.shunting_yard(regex)
            print(f"Postfix: {postfix}")
            
            # Crear AFN
            thompson = ThompsonNFA()
            nfa = thompson.regex_to_nfa(regex)
            print(f"AFN: {len(nfa.states)} estados")
            
            # Crear AFD
            subset_constructor = SubsetConstruction()
            dfa = subset_constructor.nfa_to_dfa(nfa)
            print(f"AFD: {len(dfa.states)} estados")
            
            # Minimizar AFD
            minimizer = HopcroftMinimization()
            min_dfa = minimizer.minimize_dfa(dfa)
            print(f"AFD minimizado: {len(min_dfa.states)} estados")
            
            # Guardar archivos
            filename_base = f"example_{i+1}"
            nfa.save_to_file(f"output/json/{filename_base}_nfa.json")
            dfa.save_to_file(f"output/json/{filename_base}_dfa.json")
            min_dfa.save_to_file(f"output/json/{filename_base}_min_dfa.json")
            print("Archivos JSON guardados")
            
            # Probar cadenas
            simulator = DFASimulator()
            print(f"\nPruebas con cadenas:")
            for test_string in test_strings[i]:
                accepted, trace = simulator.simulate(min_dfa, test_string)
                result = "✅ ACEPTADA" if accepted else "❌ RECHAZADA"
                print(f"  '{test_string}': {result}")
        
        except Exception as e:
            print(f"❌ Error procesando ejemplo {i+1}: {e}")

def detailed_simulation_example():
    """Ejemplo detallado de simulación"""
    print("\n" + "="*60)
    print("EJEMPLO DETALLADO DE SIMULACIÓN")
    print("="*60)
    
    regex = "(a|b)*abb"
    test_string = "aababb"
    
    print(f"Expresión regular: {regex}")
    print(f"Cadena de prueba: {test_string}")
    
    try:
        # Crear autómata
        processor = RegexProcessor()
        thompson = ThompsonNFA()
        nfa = thompson.regex_to_nfa(regex)
        
        subset_constructor = SubsetConstruction()
        dfa = subset_constructor.nfa_to_dfa(nfa)
        
        minimizer = HopcroftMinimization()
        min_dfa = minimizer.minimize_dfa(dfa)
        
        # Simulación detallada
        simulator = DFASimulator()
        accepted, trace = simulator.simulate(min_dfa, test_string)
        simulator.print_simulation(trace, accepted, test_string)
        
        # Mostrar información del autómata
        print(f"\nInformación del AFD minimizado:")
        print(f"Estados: {sorted(min_dfa.states)}")
        print(f"Alfabeto: {sorted(min_dfa.alphabet)}")
        print(f"Estado inicial: {min_dfa.start_state}")
        print(f"Estados de aceptación: {sorted(min_dfa.accept_states)}")
        
        print(f"\nTransiciones:")
        for (from_state, symbol), to_states in sorted(min_dfa.transitions.items()):
            for to_state in to_states:
                print(f"  δ({from_state}, {symbol}) = {to_state}")
    
    except Exception as e:
        print(f"❌ Error en simulación detallada: {e}")

def step_by_step_example():
    """Ejemplo paso a paso completo"""
    print("\n" + "="*60)
    print("EJEMPLO PASO A PASO COMPLETO")
    print("="*60)
    
    regex = "ab*c"
    
    print(f"1. Expresión regular original: {regex}")
    
    try:
        # Paso 1: Shunting Yard
        processor = RegexProcessor()
        postfix = processor.shunting_yard(regex)
        print(f"2. Convertir a postfix (Shunting Yard): {postfix}")
        
        # Paso 2: Thompson (Regex -> AFN)
        thompson = ThompsonNFA()
        nfa = thompson.regex_to_nfa(regex)
        print(f"3. AFN (Thompson): {len(nfa.states)} estados")
        print(f"   Estados: {sorted(nfa.states)}")
        print(f"   Estado inicial: {nfa.start_state}")
        print(f"   Estados finales: {sorted(nfa.accept_states)}")
        print(f"   Transiciones AFN:")
        for (from_state, symbol), to_states in sorted(nfa.transitions.items()):
            for to_state in to_states:
                print(f"     δ({from_state}, {symbol}) = {to_state}")
        
        # Paso 3: Construcción de subconjuntos (AFN -> AFD)
        subset_constructor = SubsetConstruction()
        dfa = subset_constructor.nfa_to_dfa(nfa)
        print(f"\n4. AFD (Construcción de subconjuntos): {len(dfa.states)} estados")
        print(f"   Estados: {sorted(dfa.states)}")
        print(f"   Estado inicial: {dfa.start_state}")
        print(f"   Estados finales: {sorted(dfa.accept_states)}")
        print(f"   Transiciones AFD:")
        for (from_state, symbol), to_states in sorted(dfa.transitions.items()):
            for to_state in to_states:
                print(f"     δ({from_state}, {symbol}) = {to_state}")
        
        # Paso 4: Minimización (Hopcroft)
        minimizer = HopcroftMinimization()
        min_dfa = minimizer.minimize_dfa(dfa)
        print(f"\n5. AFD minimizado (Hopcroft): {len(min_dfa.states)} estados")
        print(f"   Estados: {sorted(min_dfa.states)}")
        print(f"   Estado inicial: {min_dfa.start_state}")
        print(f"   Estados finales: {sorted(min_dfa.accept_states)}")
        print(f"   Transiciones AFD minimizado:")
        for (from_state, symbol), to_states in sorted(min_dfa.transitions.items()):
            for to_state in to_states:
                print(f"     δ({from_state}, {symbol}) = {to_state}")
        
        # Paso 5: Simulación
        print(f"\n6. Simulaciones:")
        simulator = DFASimulator()
        
        test_cases = ["ac", "abc", "abbc", "abbbc", "ab", "bc", "acc"]
        
        for test_string in test_cases:
            accepted, trace = simulator.simulate(min_dfa, test_string)
            result = "✅ ACEPTADA" if accepted else "❌ RECHAZADA"
            print(f"   '{test_string}': {result}")
            
            # Mostrar trace para algunas cadenas
            if test_string in ["abc", "ab"]:
                print(f"     Trace:")
                for j, (from_state, symbol, to_state) in enumerate(trace):
                    print(f"       Paso {j+1}: {from_state} --({symbol})--> {to_state}")
                if not trace:
                    print(f"       Sin transiciones válidas")
    
    except Exception as e:
        print(f"❌ Error en ejemplo paso a paso: {e}")

if __name__ == "__main__":
    print("Ejecutando ejemplos de prueba...")
    
    # Ejecutar ejemplos básicos
    test_examples()
    
    # Ejecutar simulación detallada
    detailed_simulation_example()
    
    # Ejecutar ejemplo paso a paso
    step_by_step_example()
    
    print(f"\n{'='*60}")
    print("EJEMPLOS COMPLETADOS")
    print(f"{'='*60}")
    print("Los archivos de salida incluyen:")
    print("- example_X_nfa.json, example_X_dfa.json, example_X_min_dfa.json")
    print("\nPara ejecutar el programa principal, usa:")
    print("python run_project.py")
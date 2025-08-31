#!/usr/bin/env python3
"""
Script de configuración para el proyecto de Teoría de la Computación
Estructura: src/, output/json/, run_project.py
"""

import os
import sys
import subprocess

def check_python_version():
    """Verifica que la versión de Python sea compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Error: Se requiere Python 3.7 o superior")
        print(f"Versión actual: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")

def install_dependencies():
    """Instala las dependencias opcionales"""
    print("\n📦 Instalando dependencias opcionales...")
    
    dependencies = [
        "matplotlib>=3.5.0",
        "networkx>=2.8.0"
    ]
    
    try:
        for dep in dependencies:
            print(f"Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        
        print("✅ Dependencias opcionales instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error instalando dependencias opcionales: {e}")
        print("💡 El proyecto funcionará solo con visualización en texto")
        return False

def create_directory_structure():
    """Crea la estructura de directorios necesaria"""
    directories = [
        "src",
        "output",
        "output/json"
    ]
    
    print("\n📁 Verificando estructura de directorios...")
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Directorio '{directory}' creado")
        else:
            print(f"📁 Directorio '{directory}' ya existe")

def check_required_files():
    """Verifica que los archivos principales existan"""
    required_files = [
        "src/__init__.py",
        "src/automata_project.py", 
        "run_project.py",
        "requirements.txt"
    ]
    
    print("\n📄 Verificando archivos principales...")
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - FALTANTE")
            all_exist = False
    
    return all_exist

def run_basic_test():
    """Ejecuta una prueba básica del sistema"""
    print("\n🧪 Ejecutando prueba básica...")
    
    try:
        # Añadir src al path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # Importar y probar módulos principales
        from src.automata_project import RegexProcessor, ThompsonNFA, SubsetConstruction, HopcroftMinimization, DFASimulator
        
        # Prueba simple
        regex = "ab*"
        
        processor = RegexProcessor()
        postfix = processor.shunting_yard(regex)
        
        thompson = ThompsonNFA()
        nfa = thompson.regex_to_nfa(regex)
        
        subset_constructor = SubsetConstruction()
        dfa = subset_constructor.nfa_to_dfa(nfa)
        
        minimizer = HopcroftMinimization()
        min_dfa = minimizer.minimize_dfa(dfa)
        
        simulator = DFASimulator()
        accepted, trace = simulator.simulate(min_dfa, "abb")
        
        print(f"✅ Prueba exitosa:")
        print(f"   Regex: {regex} → Postfix: {postfix}")
        print(f"   AFN: {len(nfa.states)} → AFD: {len(dfa.states)} → AFD min: {len(min_dfa.states)} estados")
        print(f"   Simulación 'abb': {'✅ Aceptada' if accepted else '❌ Rechazada'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Verifica que src/automata_project.py esté completo")
        return False
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        return False

def check_visualization_support():
    """Verifica soporte para visualizaciones"""
    print("\n🎨 Verificando soporte de visualización...")
    
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        print("✅ matplotlib + networkx disponibles")
        print("✅ Visualizaciones gráficas habilitadas")
        return True
    except ImportError:
        print("⚠️  matplotlib/networkx no disponibles")
        print("✅ Visualizaciones en texto habilitadas (suficiente)")
        return False

def print_usage_instructions():
    """Muestra instrucciones de uso"""
    print("\n" + "="*60)
    print("🎉 PROYECTO CONFIGURADO CORRECTAMENTE")
    print("="*60)
    print("\n🚀 Comandos principales:")
    print("\n1️⃣ Programa interactivo:")
    print("   python run_project.py → Opción 1")
    print("\n2️⃣ Ejemplos automáticos:")
    print("   python run_project.py → Opción 2")
    print("\n📁 Archivos clave:")
    print("   📄 src/automata_project.py - Todos los algoritmos")
    print("   📄 run_project.py - Script principal")
    print("   📁 output/json/ - Autómatas generados")
    print("\n🎯 Expresiones regulares soportadas:")
    print("   • (a|b)*abb(a|b)* - Contiene 'abb'")
    print("   • a*b+ - Cero+ 'a', una+ 'b'")
    print("   • ab*c - 'a', cero+ 'b', 'c'")
    print("   • (a|b)* - Cualquier 'a' y 'b'")
    print("\n📦 Para entregar:")

def main():
    """Función principal del script de configuración"""
    print("🛠️  CONFIGURACIÓN PROYECTO TEORÍA DE LA COMPUTACIÓN")
    print("="*60)
    
    # Verificaciones básicas
    check_python_version()
    
    if not check_required_files():
        print("\n❌ Faltan archivos principales.")
        print("💡 Asegúrate de tener la estructura completa del proyecto")
        return
    
    # Configurar estructura
    create_directory_structure()
    
    # Instalar dependencias opcionales
    deps_installed = install_dependencies()
    
    # Verificar visualizaciones
    viz_available = check_visualization_support()
    
    # Probar funcionamiento
    test_passed = run_basic_test()
    
    # Resumen final
    print("\n" + "="*60)
    print("📋 RESUMEN DE CONFIGURACIÓN")
    print("="*60)
    print(f"✅ Archivos del proyecto: OK")
    print(f"✅ Estructura de directorios: OK")
    print(f"{'✅' if deps_installed else '⚠️ '} Dependencias opcionales: {'Instaladas' if deps_installed else 'No instaladas'}")
    print(f"{'✅' if viz_available else '✅'} Visualizaciones: {'Gráficas disponibles' if viz_available else 'Solo texto (suficiente)'}")
    print(f"{'✅' if test_passed else '❌'} Funcionamiento: {'OK' if test_passed else 'ERROR'}")
    
    if test_passed:
        print_usage_instructions()
    else:
        print("\n❌ Error en las pruebas básicas.")
        print("💡 Revisa que src/automata_project.py esté completo.")

if __name__ == "__main__":
    main()
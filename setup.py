#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py híbrido para:
 - ejecutar comprobaciones / instalar deps cuando se ejecuta sin args: `python setup.py`
 - permitir empaquetado estándar con setuptools: `python setup.py sdist bdist_wheel install`
 - comprobar disponibilidad de Graphviz (binario `dot`) y requisitos Python
Ajusta los valores (name, version, entry_points) según convenga.
"""

from __future__ import annotations
import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
from typing import List, Optional

# ---------------------------
# Config (ajusta si es necesario)
# ---------------------------
PACKAGE_NAME = "proyecto1_teoria"
VERSION = "1.0.0"
DESCRIPTION = "Implementación de autómatas finitos (regexp -> NFA -> DFA -> minDFA) y utilidades."
PYTHON_REQUIRES = ">=3.8"
SRC_DIR = "src"   # carpeta donde está tu paquete
ENTRY_POINT = "autproj=src.example_usage:main"  # modificar si el módulo/función difiere
README_FILE = "README.md"
REQUIREMENTS_FILE = "requirements.txt"

# ---------------------------
# Utilidades
# ---------------------------
def read_requirements(path: str = REQUIREMENTS_FILE) -> List[str]:
    reqs = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                reqs.append(line)
    return reqs

def install_python_requirements(reqs: Optional[List[str]] = None) -> bool:
    """Instala los requisitos pip usando 'python -m pip install ...'. Devuelve True si todo OK."""
    if reqs is None:
        reqs = read_requirements()
    if not reqs:
        print("No requirements found (requirements.txt vacío o inexistente).")
        return True
    cmd = [sys.executable, "-m", "pip", "install"] + reqs
    print("Instalando dependencias python:", " ".join(reqs))
    try:
        subprocess.check_call(cmd)
        print("Dependencias Python instaladas correctamente.")
        return True
    except subprocess.CalledProcessError as e:
        print("Error al instalar dependencias Python:", e)
        return False

def is_dot_available() -> bool:
    """Comprueba si el binario 'dot' (Graphviz) está accesible en PATH."""
    # Primero comprobar con shutil.which
    if shutil.which("dot"):
        return True
    # Intentar ejecuciones directas por si están en PATH no estándar (menos probable)
    try:
        out = subprocess.run(["dot", "-V"], capture_output=True, text=True)
        # dot devuelve su versión por stderr generalmente; si returncode == 0 o 1 no es grave
        return out.returncode == 0 or out.returncode == 1 or ("dot - graphviz" in out.stderr.lower() or "graphviz" in out.stderr.lower() or "graphviz" in out.stdout.lower())
    except Exception:
        return False

def print_graphviz_install_instructions() -> None:
    system = platform.system().lower()
    print("\n--- Graphviz (dot) no encontrado en PATH ---")
    print("Para que la generación de imágenes funcione, instala Graphviz (el binario 'dot').")
    if "linux" in system:
        print("Ejemplos (según tu distro):")
        print("  Debian/Ubuntu: sudo apt update && sudo apt install -y graphviz")
        print("  Fedora: sudo dnf install -y graphviz")
        print("  Arch: sudo pacman -S graphviz")
    elif "darwin" in system or "mac" in system:
        print("macOS: brew install graphviz  (requiere Homebrew)")
    elif "windows" in system or "win" in system:
        print("Windows: descargar e instalar desde https://graphviz.org/download/ o usar Chocolatey:")
        print("  choco install graphviz")
        print("Asegúrate de añadir la carpeta 'bin' de Graphviz a tu PATH (p. ej. C:\\Program Files\\Graphviz\\bin).")
    else:
        print("Instala Graphviz desde https://graphviz.org/download/ y añade 'dot' al PATH.")
    print("Después de instalar, verifica con: dot -V\n")

def try_auto_install_graphviz() -> bool:
    """
    Intento no intrusivo de instalar Graphviz por el script (solo sugiere y, si se detecta un
    paquete gestor sin privilegios, no ejecuta instalaciones automáticas sin confirmar).
    DEV: no ejecuta instalaciones automáticas que requieran sudo sin confirmación.
    """
    system = platform.system().lower()
    # No ejecutar instalaciones automáticas por defecto (demasiado intrusivo).
    # En su lugar, mostramos instrucciones y, si el usuario pasa --auto-install-graphviz, intentamos.
    if "--auto-install-graphviz" not in sys.argv:
        return False

    print("Intentando instalación automática de Graphviz (modo --auto-install-graphviz)...")
    if "linux" in system:
        # intentar detectar gestor
        if shutil.which("apt"):
            cmd = ["sudo", "apt", "update", "&&", "sudo", "apt", "install", "-y", "graphviz"]
            print("Ejecuta manualmente (recomendado): sudo apt update && sudo apt install -y graphviz")
            return False
        if shutil.which("dnf"):
            print("Ejecuta manualmente: sudo dnf install -y graphviz")
            return False
        if shutil.which("pacman"):
            print("Ejecuta manualmente: sudo pacman -S graphviz")
            return False
    elif "darwin" in system or "mac" in system:
        if shutil.which("brew"):
            try:
                subprocess.check_call(["brew", "install", "graphviz"])
                return is_dot_available()
            except Exception as e:
                print("Error instalando con brew:", e)
                return False
    elif "windows" in system or "win" in system:
        print("En Windows intenta: choco install graphviz  (si tienes Chocolatey) o instala desde https://graphviz.org/download/")
        return False
    else:
        print("Instalación automática no soportada para este sistema desde el script.")
        return False

def run_examples() -> None:
    """Ejecuta scripts de ejemplo (si existen) para verificar la ejecución del proyecto."""
    print("\nEjecutando ejemplos de uso (si existen)...")
    # Intentamos ejecutar un archivo común de ejemplo
    candidates = [
        os.path.join(SRC_DIR, "example_usage.py"),
        os.path.join(SRC_DIR, "run_project.py"),
        os.path.join(SRC_DIR, "__main__.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Ejecutando ejemplo: {c}")
            try:
                subprocess.check_call([sys.executable, c])
            except subprocess.CalledProcessError as e:
                print(f"El ejemplo {c} devolvió error (esto puede ser normal si requiere dot):", e)
            return
    print("No se encontró un script de ejemplo estándar en 'src/'. Omite este paso.\n")

# ---------------------------
# Función principal de comportamiento por defecto (python setup.py)
# ---------------------------
def main_preparation() -> None:
    print(f"{PACKAGE_NAME} - script de preparación / comprobación\n")
    # 1) Instalar requisitos python
    reqs = read_requirements()
    if reqs:
        ok = install_python_requirements(reqs)
        if not ok:
            print("Continuando a pesar de errores en instalación de dependencias Python. Puedes instalar manualmente y volver a ejecutar.")
    else:
        print("No hay requirements o está vacío; se asume que dependencias necesarias ya están instaladas.")

    # 2) Comprobar Graphviz 'dot'
    if is_dot_available():
        print("Graphviz (dot) detectado en PATH. Visualización disponible.")
    else:
        print("Graphviz (dot) NO detectado.")
        print_graphviz_install_instructions()
        # Intento opcional si usuario indicó --auto-install-graphviz
        if "--auto-install-graphviz" in sys.argv:
            try_auto_install_graphviz()

    # 3) Probar una importación rápida del paquete para detectar errores de sintaxis
    print("\nComprobando import básico del paquete/ módulos (import)...")
    # Añadir src al path para importar en modo desarrollo
    src_path = os.path.abspath(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        # intentar importar un módulo de uso común; ajustar nombre si necesario
        import importlib
        # Si tu paquete principal tiene otro nombre, cámbialo aquí
        # Buscamos 'example_usage' o '__init__' como prueba
        if os.path.exists(os.path.join(SRC_DIR, "example_usage.py")):
            importlib.import_module("example_usage")
        else:
            # intentar importar paquete raíz
            pkg_names = [n for n in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, n)) and os.path.exists(os.path.join(SRC_DIR, n, "__init__.py"))]
            if pkg_names:
                importlib.import_module(pkg_names[0])
        print("Importación básica OK (no se detectaron errores de sintaxis al importar).")
    except Exception as e:
        print("Error al importar módulos (posible error de sintaxis o dependencias faltantes):", e)

    # 4) Ejecutar ejemplos si el usuario lo desea (opcional)
    if "--run-examples" in sys.argv:
        run_examples()
    else:
        print("\nSi deseas ejecutar ejemplos de inmediato usa: python setup.py --run-examples")
    print("\nPreparación finalizada. Si vas a empaquetar/instalar con setuptools, ejecuta: python setup.py sdist bdist_wheel install\n")

# ---------------------------
# setuptools wrapper (para empaquetado normal)
# ---------------------------
def _do_setuptools_setup() -> None:
    try:
        from setuptools import setup, find_packages  # type: ignore
    except Exception:
        print("setuptools no disponible. Instala setuptools y wheel: python -m pip install setuptools wheel")
        sys.exit(1)

    requirements = read_requirements()
    long_description = ""
    if os.path.exists(README_FILE):
        with open(README_FILE, "r", encoding="utf-8") as f:
            long_description = f.read()

    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown" if long_description else None,
        package_dir={"": SRC_DIR},
        packages=find_packages(where=SRC_DIR),
        include_package_data=True,
        install_requires=requirements,
        python_requires=PYTHON_REQUIRES,
        entry_points={"console_scripts": [ENTRY_POINT]},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        # metadata adicional opcional:
        author="Autor",
        license="MIT",
    )

# ---------------------------
# Ejecutable
# ---------------------------
if __name__ == "__main__":
    # Si hay argumentos indicativos de empaquetado clásico, usar setuptools.setup
    packaging_commands = {"sdist", "bdist_wheel", "bdist", "install", "develop", "wheel"}
    if len(sys.argv) > 1 and sys.argv[1] in packaging_commands:
        _do_setuptools_setup()
    else:
        main_preparation()

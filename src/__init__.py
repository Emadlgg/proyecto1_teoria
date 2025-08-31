"""
Proyecto Teoría de la Computación 2025
Implementación de Algoritmos para Construcción de Autómatas Finitos

Autor: [Tu Nombre]
Fecha: Agosto 2025
"""

__version__ = "1.0.0"

# Importaciones principales para facilitar el uso
from .automata_project import (
    Automata,
    RegexProcessor, 
    ThompsonNFA,
    SubsetConstruction,
    HopcroftMinimization,
    DFASimulator
)

__all__ = [
    'Automata',
    'RegexProcessor',
    'ThompsonNFA', 
    'SubsetConstruction',
    'HopcroftMinimization',
    'DFASimulator'
]
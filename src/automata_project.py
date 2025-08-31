import json
import re
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque
import os

# Visualización con matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import networkx as nx
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib/networkx no disponible. Usando solo visualización ASCII.")

class Automata:
    """Clase base para representar autómatas finitos"""
    
    def __init__(self):
        self.states: Set[int] = set()
        self.alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
        self.start_state: int = 0
        self.accept_states: Set[int] = set()
        self.epsilon = 'ε'  # Símbolo para epsilon
    
    def add_state(self, state: int) -> None:
        """Añade un estado al autómata"""
        self.states.add(state)
    
    def add_transition(self, from_state: int, symbol: str, to_state: int) -> None:
        """Añade una transición al autómata"""
        self.transitions[(from_state, symbol)].add(to_state)
        self.states.add(from_state)
        self.states.add(to_state)
        if symbol != self.epsilon:
            self.alphabet.add(symbol)
    
    def set_start_state(self, state: int) -> None:
        """Establece el estado inicial"""
        self.start_state = state
        self.states.add(state)
    
    def add_accept_state(self, state: int) -> None:
        """Añade un estado de aceptación"""
        self.accept_states.add(state)
        self.states.add(state)
    
    def to_dict(self) -> Dict:
        """Convierte el autómata a un diccionario para exportar"""
        transitions_list = []
        for (from_state, symbol), to_states in self.transitions.items():
            for to_state in to_states:
                transitions_list.append((from_state, symbol, to_state))
        
        return {
            "ESTADOS": sorted(list(self.states)),
            "SIMBOLOS": sorted(list(self.alphabet)),
            "INICIO": [self.start_state],
            "ACEPTACION": sorted(list(self.accept_states)),
            "TRANSICIONES": transitions_list
        }
    
    def save_to_file(self, filename: str) -> None:
        """Guarda el autómata en un archivo JSON"""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def visualize(self, filename: str = None, title: str = "Autómata", show_ascii: bool = True):
        """Visualiza el autómata (matplotlib si está disponible, sino ASCII)"""
        if MATPLOTLIB_AVAILABLE and filename:
            try:
                self._visualize_matplotlib(filename, title)
                return
            except Exception as e:
                print(f"Error con matplotlib: {e}")
        
        if show_ascii:
            self._visualize_ascii(title)
    
    def _visualize_matplotlib(self, filename: str, title: str):
        """Visualiza usando matplotlib y networkx"""
        # Crear grafo dirigido
        G = nx.DiGraph()
        
        # Añadir nodos
        for state in self.states:
            G.add_node(state)
        
        # Añadir aristas con etiquetas
        edge_labels = {}
        for (from_state, symbol), to_states in self.transitions.items():
            for to_state in to_states:
                if G.has_edge(from_state, to_state):
                    # Si ya existe una arista, añadir símbolo a la etiqueta
                    existing_label = edge_labels.get((from_state, to_state), "")
                    edge_labels[(from_state, to_state)] = f"{existing_label},{symbol}" if existing_label else symbol
                else:
                    G.add_edge(from_state, to_state)
                    edge_labels[(from_state, to_state)] = symbol
        
        # Configurar layout
        plt.figure(figsize=(12, 8))
        
        # Usar layout apropiado según el tamaño
        if len(self.states) <= 6:
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Dibujar nodos normales
        normal_nodes = [state for state in self.states if state not in self.accept_states]
        if normal_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, 
                                 node_color='lightblue', 
                                 node_size=1500,
                                 alpha=0.8)
        
        # Dibujar nodos de aceptación
        if self.accept_states:
            nx.draw_networkx_nodes(G, pos, nodelist=list(self.accept_states),
                                 node_color='lightgreen',
                                 node_size=1500,
                                 alpha=0.8)
            
            # Añadir círculo exterior para estados de aceptación
            for state in self.accept_states:
                x, y = pos[state]
                circle = plt.Circle((x, y), 0.12, fill=False, color='darkgreen', linewidth=2)
                plt.gca().add_patch(circle)
        
        # Destacar estado inicial con flecha
        if self.start_state in pos:
            start_x, start_y = pos[self.start_state]
            arrow = patches.FancyArrowPatch(
                (start_x - 0.3, start_y), (start_x - 0.15, start_y),
                arrowstyle='->', mutation_scale=20, color='red', linewidth=3
            )
            plt.gca().add_patch(arrow)
            plt.text(start_x - 0.4, start_y, 'INICIO', fontsize=10, color='red', fontweight='bold')
        
        # Dibujar aristas
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                             arrowsize=20, arrowstyle='->', alpha=0.7,
                             connectionstyle="arc3,rad=0.15", width=1.5)
        
        # Dibujar etiquetas de nodos
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
        
        # Dibujar etiquetas de aristas
        for (u, v), label in edge_labels.items():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            # Calcular punto medio con offset
            mid_x = (x1 + x2) / 2 + 0.1
            mid_y = (y1 + y2) / 2 + 0.1
            
            plt.text(mid_x, mid_y, label, fontsize=11, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Configurar plot
        plt.title(f"{title}\nEstados: {len(self.states)} | Inicial: {self.start_state} | Finales: {sorted(self.accept_states)}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Añadir leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=12, label='Estado normal'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=12, label='Estado final'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Estado inicial')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Guardar archivo
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Visualización guardada como '{filename}.png'")
        
        plt.close()  # Cerrar para liberar memoria
    
    def _visualize_ascii(self, title: str = "Autómata"):
        """Visualiza el autómata en ASCII"""
        print(f"\n{'='*60}")
        print(f"📊 {title.upper()}")
        print(f"{'='*60}")
        
        # Información básica
        print(f"🔹 Estados: {sorted(self.states)}")
        print(f"🔹 Alfabeto: {sorted(self.alphabet)}")
        print(f"🔹 Estado inicial: {self.start_state}")
        print(f"🔹 Estados finales: {sorted(self.accept_states)}")
        
        # Tabla de transiciones
        print(f"\n📋 TABLA DE TRANSICIONES:")
        print("┌─────────┬─────────┬─────────┐")
        print("│ Estado  │ Símbolo │ Destino │")
        print("├─────────┼─────────┼─────────┤")
        
        # Ordenar transiciones para mejor legibilidad
        transitions = []
        for (from_state, symbol), to_states in self.transitions.items():
            for to_state in to_states:
                transitions.append((from_state, symbol, to_state))
        
        transitions.sort()
        
        for from_state, symbol, to_state in transitions:
            state_marker = "→" if from_state == self.start_state else " "
            final_marker = "✓" if to_state in self.accept_states else " "
            symbol_display = symbol if symbol != self.epsilon else "ε"
            
            print(f"│ {state_marker}{from_state:^6} │ {symbol_display:^7} │ {to_state:^6}{final_marker} │")
        
        print("└─────────┴─────────┴─────────┘")
        
        # Diagrama simple
        print(f"\n🔗 DIAGRAMA SIMPLIFICADO:")
        states = sorted(self.states)
        
        # Línea de estados
        state_line = ""
        for state in states:
            if state == self.start_state and state in self.accept_states:
                marker = f"→(({state}))"  # Inicial Y final
            elif state == self.start_state:
                marker = f"→({state})"    # Solo inicial
            elif state in self.accept_states:
                marker = f"(({state}))"   # Solo final
            else:
                marker = f"({state})"     # Normal
            state_line += f"{marker:^10}"
        
        print(state_line)
        
        # Mostrar transiciones principales (no-epsilon)
        print(f"\n🔀 TRANSICIONES:")
        for (from_state, symbol), to_states in sorted(self.transitions.items()):
            if symbol != self.epsilon:
                for to_state in to_states:
                    print(f"   {from_state} ──{symbol}──> {to_state}")

class RegexProcessor:
    """Procesador de expresiones regulares"""
    
    def __init__(self):
        self.operators = {'|': 1, '+': 2, '*': 3}
        self.right_associative = set()
    
    def shunting_yard(self, regex: str) -> str:
        """Convierte una expresión regular a notación postfix usando el algoritmo Shunting Yard"""
        # Preprocesar para añadir operadores de concatenación implícitos
        preprocessed = self._add_concatenation_operators(regex)
        
        output = []
        operator_stack = []
        
        for char in preprocessed:
            if char.isalnum() or char == 'ε':
                output.append(char)
            elif char == '(':
                operator_stack.append(char)
            elif char == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                if operator_stack:
                    operator_stack.pop()  # Remove '('
            elif char in self.operators:
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       operator_stack[-1] in self.operators and
                       (self.operators[operator_stack[-1]] > self.operators[char] or
                        (self.operators[operator_stack[-1]] == self.operators[char] and
                         char not in self.right_associative))):
                    output.append(operator_stack.pop())
                operator_stack.append(char)
        
        while operator_stack:
            output.append(operator_stack.pop())
        
        return ''.join(output)
    
    def _add_concatenation_operators(self, regex: str) -> str:
        """Añade operadores de concatenación explícitos donde sea necesario"""
        result = []
        
        for i, char in enumerate(regex):
            result.append(char)
            
            if i < len(regex) - 1:
                next_char = regex[i + 1]
                
                # Añadir '+' para concatenación entre:
                if ((char.isalnum() or char == 'ε') and (next_char.isalnum() or next_char == 'ε' or next_char == '(')) or \
                   (char == ')' and (next_char.isalnum() or next_char == 'ε' or next_char == '(')) or \
                   (char == '*' and (next_char.isalnum() or next_char == 'ε' or next_char == '(')):
                    result.append('+')
        
        return ''.join(result)

class ThompsonNFA:
    """Implementación del algoritmo de Thompson para convertir regex a AFN"""
    
    def __init__(self):
        self.state_counter = 0
    
    def get_next_state(self) -> int:
        """Obtiene el siguiente estado disponible"""
        state = self.state_counter
        self.state_counter += 1
        return state
    
    def regex_to_nfa(self, regex: str) -> Automata:
        """Convierte una expresión regular a AFN usando el algoritmo de Thompson"""
        processor = RegexProcessor()
        postfix = processor.shunting_yard(regex)
        
        stack = []
        
        for char in postfix:
            if char.isalnum() or char == 'ε':
                # Símbolo básico
                nfa = self._basic_nfa(char)
                stack.append(nfa)
            elif char == '+':
                # Concatenación
                if len(stack) >= 2:
                    nfa2 = stack.pop()
                    nfa1 = stack.pop()
                    nfa = self._concatenate(nfa1, nfa2)
                    stack.append(nfa)
            elif char == '|':
                # Unión
                if len(stack) >= 2:
                    nfa2 = stack.pop()
                    nfa1 = stack.pop()
                    nfa = self._union(nfa1, nfa2)
                    stack.append(nfa)
            elif char == '*':
                # Estrella de Kleene
                if len(stack) >= 1:
                    nfa1 = stack.pop()
                    nfa = self._kleene_star(nfa1)
                    stack.append(nfa)
        
        return stack[0] if stack else Automata()
    
    def _basic_nfa(self, symbol: str) -> Automata:
        """Crea un AFN básico para un símbolo"""
        nfa = Automata()
        start = self.get_next_state()
        end = self.get_next_state()
        
        nfa.set_start_state(start)
        nfa.add_accept_state(end)
        nfa.add_transition(start, symbol, end)
        
        return nfa
    
    def _concatenate(self, nfa1: Automata, nfa2: Automata) -> Automata:
        """Concatena dos AFN"""
        nfa = Automata()
        
        # Copiar estados y transiciones de ambos AFN
        for state in nfa1.states:
            nfa.add_state(state)
        for state in nfa2.states:
            nfa.add_state(state)
            
        for (from_state, symbol), to_states in nfa1.transitions.items():
            for to_state in to_states:
                nfa.add_transition(from_state, symbol, to_state)
                
        for (from_state, symbol), to_states in nfa2.transitions.items():
            for to_state in to_states:
                nfa.add_transition(from_state, symbol, to_state)
        
        # Conectar estados de aceptación de nfa1 con estado inicial de nfa2
        for accept_state in nfa1.accept_states:
            nfa.add_transition(accept_state, nfa.epsilon, nfa2.start_state)
        
        nfa.set_start_state(nfa1.start_state)
        nfa.accept_states = nfa2.accept_states.copy()
        
        return nfa
    
    def _union(self, nfa1: Automata, nfa2: Automata) -> Automata:
        """Une dos AFN"""
        nfa = Automata()
        
        # Nuevo estado inicial y final
        new_start = self.get_next_state()
        new_end = self.get_next_state()
        
        # Copiar estados y transiciones
        nfa.add_state(new_start)
        nfa.add_state(new_end)
        
        for state in nfa1.states:
            nfa.add_state(state)
        for state in nfa2.states:
            nfa.add_state(state)
            
        for (from_state, symbol), to_states in nfa1.transitions.items():
            for to_state in to_states:
                nfa.add_transition(from_state, symbol, to_state)
                
        for (from_state, symbol), to_states in nfa2.transitions.items():
            for to_state in to_states:
                nfa.add_transition(from_state, symbol, to_state)
        
        # Conectar nuevo inicio con inicios de ambos AFN
        nfa.add_transition(new_start, nfa.epsilon, nfa1.start_state)
        nfa.add_transition(new_start, nfa.epsilon, nfa2.start_state)
        
        # Conectar finales de ambos AFN con nuevo final
        for accept_state in nfa1.accept_states:
            nfa.add_transition(accept_state, nfa.epsilon, new_end)
        for accept_state in nfa2.accept_states:
            nfa.add_transition(accept_state, nfa.epsilon, new_end)
        
        nfa.set_start_state(new_start)
        nfa.add_accept_state(new_end)
        
        return nfa
    
    def _kleene_star(self, nfa1: Automata) -> Automata:
        """Aplica la estrella de Kleene a un AFN"""
        nfa = Automata()
        
        # Nuevo estado inicial y final
        new_start = self.get_next_state()
        new_end = self.get_next_state()
        
        # Copiar estados y transiciones
        nfa.add_state(new_start)
        nfa.add_state(new_end)
        
        for state in nfa1.states:
            nfa.add_state(state)
            
        for (from_state, symbol), to_states in nfa1.transitions.items():
            for to_state in to_states:
                nfa.add_transition(from_state, symbol, to_state)
        
        # Conexiones para la estrella de Kleene
        nfa.add_transition(new_start, nfa.epsilon, nfa1.start_state)  # Entrada
        nfa.add_transition(new_start, nfa.epsilon, new_end)  # Cadena vacía
        
        for accept_state in nfa1.accept_states:
            nfa.add_transition(accept_state, nfa.epsilon, new_end)  # Salida
            nfa.add_transition(accept_state, nfa.epsilon, nfa1.start_state)  # Repetición
        
        nfa.set_start_state(new_start)
        nfa.add_accept_state(new_end)
        
        return nfa

class SubsetConstruction:
    """Implementación de la construcción de subconjuntos para convertir AFN a AFD"""
    
    def nfa_to_dfa(self, nfa: Automata) -> Automata:
        """Convierte un AFN a AFD usando construcción de subconjuntos"""
        dfa = Automata()
        
        # Calcular epsilon-closure del estado inicial
        start_closure = self._epsilon_closure(nfa, {nfa.start_state})
        
        # Mapeo de conjuntos de estados AFN a estados AFD
        state_mapping = {}
        state_counter = 0
        
        # Cola para procesar estados
        unprocessed = deque()
        
        # Procesar estado inicial
        start_key = frozenset(start_closure)
        state_mapping[start_key] = state_counter
        dfa.set_start_state(state_counter)
        
        if start_closure.intersection(nfa.accept_states):
            dfa.add_accept_state(state_counter)
        
        unprocessed.append(start_closure)
        state_counter += 1
        
        # Procesar todos los estados
        while unprocessed:
            current_set = unprocessed.popleft()
            current_key = frozenset(current_set)
            current_state = state_mapping[current_key]
            
            # Para cada símbolo del alfabeto
            for symbol in nfa.alphabet:
                # Calcular el conjunto de estados alcanzables
                next_set = set()
                for state in current_set:
                    if (state, symbol) in nfa.transitions:
                        next_set.update(nfa.transitions[(state, symbol)])
                
                if next_set:
                    # Aplicar epsilon-closure
                    next_closure = self._epsilon_closure(nfa, next_set)
                    next_key = frozenset(next_closure)
                    
                    # Si es un nuevo estado, añadirlo
                    if next_key not in state_mapping:
                        state_mapping[next_key] = state_counter
                        
                        if next_closure.intersection(nfa.accept_states):
                            dfa.add_accept_state(state_counter)
                        
                        unprocessed.append(next_closure)
                        state_counter += 1
                    
                    # Añadir transición
                    dfa.add_transition(current_state, symbol, state_mapping[next_key])
        
        return dfa
    
    def _epsilon_closure(self, nfa: Automata, states: Set[int]) -> Set[int]:
        """Calcula la epsilon-clausura de un conjunto de estados"""
        closure = set(states)
        stack = list(states)
        
        while stack:
            state = stack.pop()
            if (state, nfa.epsilon) in nfa.transitions:
                for next_state in nfa.transitions[(state, nfa.epsilon)]:
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)
        
        return closure

class HopcroftMinimization:
    """Implementación del algoritmo de minimización de Hopcroft"""
    
    def minimize_dfa(self, dfa: Automata) -> Automata:
        """Minimiza un AFD usando el algoritmo de Hopcroft"""
        # Inicializar particiones
        accept_states = dfa.accept_states
        non_accept_states = dfa.states - accept_states
        
        partitions = []
        if non_accept_states:
            partitions.append(non_accept_states)
        if accept_states:
            partitions.append(accept_states)
        
        # Refinar particiones
        changed = True
        while changed:
            changed = False
            new_partitions = []
            
            for partition in partitions:
                refined = self._refine_partition(dfa, partition, partitions)
                if len(refined) > 1:
                    changed = True
                new_partitions.extend(refined)
            
            partitions = new_partitions
        
        # Construir AFD minimizado
        return self._build_minimized_dfa(dfa, partitions)
    
    def _refine_partition(self, dfa: Automata, partition: Set[int], all_partitions: List[Set[int]]) -> List[Set[int]]:
        """Refina una partición basada en las transiciones"""
        if len(partition) <= 1:
            return [partition]
        
        # Agrupar estados por su comportamiento de transición
        groups = defaultdict(set)
        
        for state in partition:
            signature = []
            for symbol in sorted(dfa.alphabet):
                target_partition = None
                if (state, symbol) in dfa.transitions:
                    target_state = next(iter(dfa.transitions[(state, symbol)]))
                    for i, p in enumerate(all_partitions):
                        if target_state in p:
                            target_partition = i
                            break
                signature.append(target_partition)
            
            groups[tuple(signature)].add(state)
        
        return list(groups.values())
    
    def _build_minimized_dfa(self, dfa: Automata, partitions: List[Set[int]]) -> Automata:
        """Construye el AFD minimizado a partir de las particiones"""
        min_dfa = Automata()
        
        # Mapear particiones a nuevos estados
        partition_to_state = {}
        for i, partition in enumerate(partitions):
            partition_to_state[frozenset(partition)] = i
        
        # Encontrar estado inicial y estados de aceptación
        for i, partition in enumerate(partitions):
            min_dfa.add_state(i)
            
            if dfa.start_state in partition:
                min_dfa.set_start_state(i)
            
            if partition.intersection(dfa.accept_states):
                min_dfa.add_accept_state(i)
        
        # Construir transiciones
        for i, partition in enumerate(partitions):
            representative = next(iter(partition))
            
            for symbol in dfa.alphabet:
                if (representative, symbol) in dfa.transitions:
                    target = next(iter(dfa.transitions[(representative, symbol)]))
                    
                    # Encontrar partición objetivo
                    for j, target_partition in enumerate(partitions):
                        if target in target_partition:
                            min_dfa.add_transition(i, symbol, j)
                            break
        
        return min_dfa

class DFASimulator:
    """Simulador de AFD"""
    
    def simulate(self, dfa: Automata, input_string: str) -> Tuple[bool, List[Tuple[int, str, int]]]:
        """Simula la ejecución de una cadena en el AFD"""
        current_state = dfa.start_state
        trace = []
        
        for symbol in input_string:
            if symbol not in dfa.alphabet:
                return False, trace
            
            if (current_state, symbol) not in dfa.transitions:
                return False, trace
            
            next_state = next(iter(dfa.transitions[(current_state, symbol)]))
            trace.append((current_state, symbol, next_state))
            current_state = next_state
        
        accepted = current_state in dfa.accept_states
        return accepted, trace
    
    def print_simulation(self, trace: List[Tuple[int, str, int]], accepted: bool, input_string: str):
        """Imprime la simulación paso a paso"""
        print(f"\n🔄 Simulación de la cadena: '{input_string}'")
        print("─" * 50)
        
        if not trace:
            print("❌ La cadena no puede ser procesada (símbolo inválido)")
            return
        
        # Mostrar estado inicial
        if trace:
            print(f"Estado inicial: {trace[0][0]}")
        
        for i, (from_state, symbol, to_state) in enumerate(trace):
            print(f"Paso {i+1}: Estado {from_state} ──({symbol})──> Estado {to_state}")
        
        print("─" * 50)
        if accepted:
            print("✅ CADENA ACEPTADA")
        else:
            print("❌ CADENA RECHAZADA")

def main():
    """Función principal del programa"""
    print("🎯 === PROYECTO TEORÍA DE LA COMPUTACIÓN ===")
    print("Implementación de Algoritmos para Autómatas Finitos\n")
    
    # Crear directorios de salida si no existen
    os.makedirs("output/json", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)
    
    # Solicitar expresión regular
    regex = input("📝 Ingrese la expresión regular: ")
    print(f"\n🔍 Procesando expresión regular: {regex}")
    
    try:
        # 1. Convertir a postfix (Shunting Yard)
        print("\n1️⃣ ALGORITMO SHUNTING YARD")
        processor = RegexProcessor()
        postfix = processor.shunting_yard(regex)
        print(f"   Notación postfix: {postfix}")
        
        # 2. Convertir regex a AFN (Thompson)
        print("\n2️⃣ ALGORITMO DE THOMPSON (REGEX → AFN)")
        thompson = ThompsonNFA()
        nfa = thompson.regex_to_nfa(regex)
        print(f"   AFN creado con {len(nfa.states)} estados")
        
        # Guardar AFN
        nfa.save_to_file("output/json/nfa.json")
        print("   ✅ AFN guardado en 'output/json/nfa.json'")
        
        # Visualizar AFN
        nfa.visualize("output/visualizations/nfa_graph", "AFN (Thompson)")
        
        # 3. Convertir AFN a AFD (Construcción de Subconjuntos)
        print("\n3️⃣ CONSTRUCCIÓN DE SUBCONJUNTOS (AFN → AFD)")
        subset_constructor = SubsetConstruction()
        dfa = subset_constructor.nfa_to_dfa(nfa)
        print(f"   AFD creado con {len(dfa.states)} estados")
        
        # Guardar AFD
        dfa.save_to_file("output/json/dfa.json")
        print("   ✅ AFD guardado en 'output/json/dfa.json'")
        
        # Visualizar AFD
        dfa.visualize("output/visualizations/dfa_graph", "AFD (Subconjuntos)")
        
        # 4. Minimizar AFD (Hopcroft)
        print("\n4️⃣ MINIMIZACIÓN DE HOPCROFT (AFD → AFD MINIMAL)")
        minimizer = HopcroftMinimization()
        min_dfa = minimizer.minimize_dfa(dfa)
        print(f"   AFD minimizado creado con {len(min_dfa.states)} estados")
        
        # Guardar AFD minimizado
        min_dfa.save_to_file("output/json/min_dfa.json")
        print("   ✅ AFD minimizado guardado en 'output/json/min_dfa.json'")
        
        # Visualizar AFD minimizado
        min_dfa.visualize("output/visualizations/min_dfa_graph", "AFD Minimizado (Hopcroft)")
        
        # 5. Simulaciones
        print("\n5️⃣ SIMULACIONES")
        simulator = DFASimulator()
        
        print("=" * 60)
        print("🎮 SIMULADOR DE AFD")
        print("=" * 60)
        
        while True:
            test_string = input("\n📝 Ingrese una cadena para probar (o 'quit' para salir): ")
            if test_string.lower() == 'quit':
                break
            
            accepted, trace = simulator.simulate(min_dfa, test_string)
            simulator.print_simulation(trace, accepted, test_string)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Programa interrumpido por el usuario.")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        print("💡 Verifica que la expresión regular sea válida.")

if __name__ == "__main__":
    main()
"""
Módulo principal del proyecto - Autómatas finitos, regex -> AFN -> AFD -> AFD minimizado
Incluye: export JSON con epsilon como "", render Graphviz/DOT, export Cypher (Neo4j)
Fecha: Agosto 2025
"""

import json
import os
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque

try:
    from graphviz import Digraph # type: ignore
    GRAPHVIZ_AVAILABLE = True
except Exception:
    GRAPHVIZ_AVAILABLE = False

try:
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

class Automata:

    def __init__(self):
        self.states: Set[int] = set()
        self.alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
        self.start_state: Optional[int] = None
        self.accept_states: Set[int] = set()
        self.epsilon = 'ε'

    def add_state(self, state: int) -> None:
        self.states.add(state)

    def add_transition(self, from_state: int, symbol: str, to_state: int) -> None:
        self.transitions[(from_state, symbol)].add(to_state)
        self.add_state(from_state)
        self.add_state(to_state)
        if symbol != self.epsilon and symbol != "":
            self.alphabet.add(symbol)

    def set_start_state(self, state: int) -> None:
        self.start_state = state
        self.add_state(state)

    def add_accept_state(self, state: int) -> None:
        self.accept_states.add(state)
        self.add_state(state)

    def to_dict(self) -> Dict:
        transitions_list: List[List] = []
        for (from_state, symbol), to_states in self.transitions.items():
            for to_state in to_states:
                export_symbol = "" if symbol == self.epsilon else symbol
                transitions_list.append([from_state, export_symbol, to_state])

        symbols = sorted(s for s in self.alphabet if s != self.epsilon and s != "")

        return {
            "ESTADOS": sorted(list(self.states)),
            "SIMBOLOS": symbols,
            "INICIO": [self.start_state] if self.start_state is not None else [],
            "ACEPTACION": sorted(list(self.accept_states)),
            "TRANSICIONES": transitions_list
        }

    def save_to_file(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✅ Guardado JSON: {filename}")

    def visualize(self, filename: str = None, title: str = "Autómata", show_ascii: bool = True):
        if GRAPHVIZ_AVAILABLE and filename:
            try:
                out = self.render_with_graphviz(filename, title)
                print(f"✅ Visualización Graphviz: {out}")
                return
            except Exception as e:
                print(f"⚠️ Error render Graphviz: {e}  -- intentando matplotlib...")

        if MATPLOTLIB_AVAILABLE and filename:
            try:
                self._visualize_matplotlib(filename, title)
                return
            except Exception as e:
                print(f"⚠️ Error matplotlib: {e}")

        if show_ascii:
            self._visualize_ascii(title)

    def render_with_graphviz(self, filename: str, title: str = "Autómata") -> str:
        dot = Digraph(comment=title, format='png')
        dot.attr(rankdir='LR')

        for s in sorted(self.states):
            if s in self.accept_states:
                dot.node(str(s), shape='doublecircle')
            else:
                dot.node(str(s), shape='circle')

        if self.start_state is not None:
            dot.node('start_point', label='', shape='point')
            dot.edge('start_point', str(self.start_state))

        edge_labels = {}
        for (u, sym), to_states in self.transitions.items():
            for v in to_states:
                label = 'ε' if sym == self.epsilon else sym
                key = (str(u), str(v))
                if key in edge_labels:
                    edge_labels[key].append(label)
                else:
                    edge_labels[key] = [label]

        for (u, v), labels in edge_labels.items():
            lbl = ",".join(labels)
            dot.edge(u, v, label=lbl)

        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        out_path = dot.render(filename, cleanup=True)
        return out_path

    def export_dot(self, filename: str) -> None:
        lines = []
        lines.append('digraph Automata {')
        lines.append('  rankdir=LR;')
        lines.append('  node [shape = circle];')
        for s in sorted(self.states):
            shape = 'doublecircle' if s in self.accept_states else 'circle'
            lines.append(f'  {s} [shape={shape}];')
        if self.start_state is not None:
            lines.append(f'  start_point [shape=point,label=""];')
            lines.append(f'  start_point -> {self.start_state};')
        for (u, sym), to_states in self.transitions.items():
            for v in to_states:
                label = 'ε' if sym == self.epsilon else sym
                lines.append(f'  {u} -> {v} [label="{label}"];')
        lines.append('}')
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"✅ DOT exportado: {filename}")

    def export_neo4j_cypher(self, filename: str) -> None:
        lines = []
        lines.append("// Cypher export generado desde automata")
        for s in sorted(self.states):
            props = []
            props.append(f"id: {s}")
            if self.start_state == s:
                props.append("start: true")
            if s in self.accept_states:
                props.append("accept: true")
            props_str = ", ".join(props)
            lines.append(f"CREATE (:State {{{props_str}}});")
        for (u, sym), to_states in self.transitions.items():
            for v in to_states:
                label = '' if sym == self.epsilon else sym
                lines.append(f"CREATE (a:State {{id: {u}}})-[:TRANSITION {{symbol: \"{label}\"}}]->(b:State {{id: {v}}});")
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"✅ Cypher exportado: {filename}")

    def _visualize_ascii(self, title: str = "Autómata"):
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print(f"{'='*60}")
        print(f"Estados: {sorted(self.states)}")
        print(f"Alfabeto: {sorted(self.alphabet)}")
        print(f"Estado inicio: {self.start_state}")
        print(f"Estados aceptacion: {sorted(self.accept_states)}")
        print("\nTRANSICIONES:")
        trans = []
        for (u, sym), to_states in self.transitions.items():
            for v in to_states:
                symbol = 'ε' if sym == self.epsilon else sym
                trans.append((u, symbol, v))
        trans.sort()
        for u, sym, v in trans:
            print(f"  {u} --({sym})-> {v}")
        print(f"{'='*60}\n")



class RegexProcessor:
    def __init__(self):
        self.operators = {'|': 1, '+': 2, '*': 3}
        self.right_associative = set()  

    def _is_symbol(self, c: str) -> bool:
        return c.isalnum() or c == 'ε' or c == ''

    def _add_concatenation_operators(self, regex: str) -> str:
        result = []
        n = len(regex)
        for i, c in enumerate(regex):
            result.append(c)
            if i < n - 1:
                nxt = regex[i+1]
                left_is = (c.isalnum() or c == 'ε' or c == ')' or c == '*')
                right_is = (nxt.isalnum() or nxt == 'ε' or nxt == '(')
                if left_is and right_is:
                    result.append('+')
        return ''.join(result)

    def shunting_yard(self, regex: str) -> str:
        pre = self._add_concatenation_operators(regex)
        output: List[str] = []
        stack: List[str] = []
        for token in pre:
            if token.isalnum() or token == 'ε':
                output.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack and stack[-1] == '(':
                    stack.pop()
            elif token in self.operators:
                while (stack and stack[-1] != '(' and stack[-1] in self.operators and
                       (self.operators[stack[-1]] > self.operators[token] or
                        (self.operators[stack[-1]] == self.operators[token] and token not in self.right_associative))):
                    output.append(stack.pop())
                stack.append(token)
            else:
                continue
        while stack:
            output.append(stack.pop())
        return ''.join(output)



class ThompsonNFA:
    def __init__(self):
        self.state_counter = 0

    def _next_state(self) -> int:
        s = self.state_counter
        self.state_counter += 1
        return s

    def regex_to_nfa(self, regex: str) -> Automata:
        proc = RegexProcessor()
        postfix = proc.shunting_yard(regex)
        stack: List[Automata] = []

        for token in postfix:
            if token.isalnum() or token == 'ε':
                stack.append(self._basic_nfa(token))
            elif token == '+': 
                if len(stack) < 2:
                    raise ValueError("Concatenación: pila insuficiente")
                b = stack.pop()
                a = stack.pop()
                stack.append(self._concatenate(a, b))
            elif token == '|':
                if len(stack) < 2:
                    raise ValueError("Union: pila insuficiente")
                b = stack.pop()
                a = stack.pop()
                stack.append(self._union(a, b))
            elif token == '*':
                if len(stack) < 1:
                    raise ValueError("Kleene: pila insuficiente")
                a = stack.pop()
                stack.append(self._kleene_star(a))
            else:
                continue

        return stack[0] if stack else Automata()

    def _basic_nfa(self, symbol: str) -> Automata:
        nfa = Automata()
        s = self._next_state()
        t = self._next_state()
        nfa.set_start_state(s)
        nfa.add_accept_state(t)
        nfa.add_transition(s, symbol if symbol != '' else nfa.epsilon, t)
        return nfa

    def _concatenate(self, nfa1: Automata, nfa2: Automata) -> Automata:
        nfa = Automata()
        for (u, sym), to_states in list(nfa1.transitions.items()):
            for v in to_states:
                nfa.add_transition(u, sym, v)
        for (u, sym), to_states in list(nfa2.transitions.items()):
            for v in to_states:
                nfa.add_transition(u, sym, v)
        nfa.set_start_state(nfa1.start_state)
        for a in nfa1.accept_states:
            nfa.add_transition(a, nfa.epsilon, nfa2.start_state)
        for a in nfa2.accept_states:
            nfa.add_accept_state(a)
        return nfa

    def _union(self, nfa1: Automata, nfa2: Automata) -> Automata:
        nfa = Automata()
        new_start = self._next_state()
        new_end = self._next_state()
        nfa.add_state(new_start)
        nfa.add_state(new_end)
        for (u, sym), to_states in list(nfa1.transitions.items()):
            for v in to_states:
                nfa.add_transition(u, sym, v)
        for (u, sym), to_states in list(nfa2.transitions.items()):
            for v in to_states:
                nfa.add_transition(u, sym, v)
        nfa.add_transition(new_start, nfa.epsilon, nfa1.start_state)
        nfa.add_transition(new_start, nfa.epsilon, nfa2.start_state)
        for a in nfa1.accept_states:
            nfa.add_transition(a, nfa.epsilon, new_end)
        for a in nfa2.accept_states:
            nfa.add_transition(a, nfa.epsilon, new_end)
        nfa.set_start_state(new_start)
        nfa.add_accept_state(new_end)
        return nfa

    def _kleene_star(self, nfa1: Automata) -> Automata:
        nfa = Automata()
        new_start = self._next_state()
        new_end = self._next_state()
        nfa.add_state(new_start)
        nfa.add_state(new_end)
        for (u, sym), to_states in list(nfa1.transitions.items()):
            for v in to_states:
                nfa.add_transition(u, sym, v)
        nfa.add_transition(new_start, nfa.epsilon, nfa1.start_state)
        nfa.add_transition(new_start, nfa.epsilon, new_end)
        for a in nfa1.accept_states:
            nfa.add_transition(a, nfa.epsilon, new_end)
            nfa.add_transition(a, nfa.epsilon, nfa1.start_state)
        nfa.set_start_state(new_start)
        nfa.add_accept_state(new_end)
        return nfa

class SubsetConstruction:

    def _epsilon_closure(self, nfa: Automata, states: Set[int]) -> Set[int]:
        closure = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            if (s, nfa.epsilon) in nfa.transitions:
                for t in nfa.transitions[(s, nfa.epsilon)]:
                    if t not in closure:
                        closure.add(t)
                        stack.append(t)
        return closure

    def nfa_to_dfa(self, nfa: Automata, add_dead_state: bool = False, max_states: int = 10000, verbose: bool = False) -> Automata:
        from collections import deque

        dfa = Automata()

        alphabet = sorted(s for s in nfa.alphabet if s != nfa.epsilon and s != "")

        if nfa.start_state is None:
            dfa.set_start_state(0)
            return dfa

        start_closure = self._epsilon_closure(nfa, {nfa.start_state})
        start_key = frozenset(start_closure)
        state_map: Dict[frozenset, int] = {}
        queue = deque()

        state_map[start_key] = 0
        dfa.set_start_state(0)
        if start_closure & nfa.accept_states:
            dfa.add_accept_state(0)
        queue.append(start_key)
        next_state_id = 1

        visited: Set[frozenset] = set()

        while queue:
            cur_key = queue.popleft()
            if cur_key in visited:
                continue
            visited.add(cur_key)

            cur_state = state_map[cur_key]
            if verbose:
                print(f"[nfa->dfa] procesando D-state {cur_state} (conjunto size={len(cur_key)})")

            for symbol in alphabet:
                move_set = set()
                for s in cur_key:
                    move_set.update(nfa.transitions.get((s, symbol), set()))

                if move_set:
                    closure = self._epsilon_closure(nfa, move_set)
                else:
                    closure = set()

                key2 = frozenset(closure)

                if not closure:
                    if add_dead_state:
                        pass
                    else:
                        continue

                if key2 not in state_map:
                    if next_state_id > max_states:
                        raise RuntimeError(f"[nfa_to_dfa] Excedido límite de estados ({max_states}). Expresión puede causar explosión exponencial.")
                    state_map[key2] = next_state_id
                    if closure & nfa.accept_states:
                        dfa.add_accept_state(next_state_id)
                    queue.append(key2)
                    if verbose:
                        print(f"[nfa->dfa]  nuevo D-state {next_state_id} (size={len(key2)})")
                    next_state_id += 1

                dfa.add_transition(cur_state, symbol, state_map[key2])

        if add_dead_state:
            sink_id = None
            empty_key = frozenset()
            if empty_key in state_map:
                sink_id = state_map[empty_key]
            else:
                sink_id = next_state_id
                state_map[empty_key] = sink_id
                next_state_id += 1

            for s in list(state_map.values()):
                for symbol in alphabet:
                    if (s, symbol) not in dfa.transitions or not dfa.transitions.get((s, symbol)):
                        dfa.add_transition(s, symbol, sink_id)
            for symbol in alphabet:
                dfa.add_transition(sink_id, symbol, sink_id)

        return dfa


class HopcroftMinimization:
    def minimize_dfa(self, dfa: Automata) -> Automata:
        F = set(dfa.accept_states)
        Q = set(dfa.states)
        nonF = Q - F
        partitions: List[Set[int]] = []
        if nonF:
            partitions.append(nonF)
        if F:
            partitions.append(F)

        changed = True
        while changed:
            changed = False
            new_parts: List[Set[int]] = []
            for P in partitions:
                grouped = defaultdict(set)
                for state in P:
                    signature = []
                    for symbol in sorted(s for s in dfa.alphabet if s != dfa.epsilon and s != ""):
                        target = None
                        if (state, symbol) in dfa.transitions:
                            t = next(iter(dfa.transitions[(state, symbol)]))
                            idx = next((i for i, part in enumerate(partitions) if t in part), None)
                            target = idx
                        signature.append(target)
                    grouped[tuple(signature)].add(state)
                if len(grouped) == 1:
                    new_parts.append(P)
                else:
                    changed = True
                    for g in grouped.values():
                        new_parts.append(g)
            partitions = new_parts

        min_dfa = Automata()
        part_to_state = {frozenset(p): i for i, p in enumerate(partitions)}
        for i, p in enumerate(partitions):
            min_dfa.add_state(i)
            if dfa.start_state in p:
                min_dfa.set_start_state(i)
            if p.intersection(dfa.accept_states):
                min_dfa.add_accept_state(i)

        for i, p in enumerate(partitions):
            rep = next(iter(p))
            for symbol in sorted(s for s in dfa.alphabet if s != dfa.epsilon and s != ""):
                if (rep, symbol) in dfa.transitions:
                    tgt = next(iter(dfa.transitions[(rep, symbol)]))
                    tgt_part_idx = next((j for j, part in enumerate(partitions) if tgt in part), None)
                    if tgt_part_idx is not None:
                        min_dfa.add_transition(i, symbol, tgt_part_idx)

        return min_dfa

class DFASimulator:

    def simulate(self, dfa: Automata, input_string: str) -> Tuple[bool, List[Tuple[int, str, int]]]:
        if dfa.start_state is None:
            return False, []
        current = dfa.start_state
        trace: List[Tuple[int, str, int]] = []
        for ch in input_string:
            if ch not in dfa.alphabet:
                return False, trace
            if (current, ch) not in dfa.transitions:
                return False, trace
            nxt = next(iter(dfa.transitions[(current, ch)]))
            trace.append((current, ch, nxt))
            current = nxt
        accepted = current in dfa.accept_states
        return accepted, trace

    def print_simulation(self, trace: List[Tuple[int, str, int]], accepted: bool, input_string: str) -> None:
        print(f"\nSimulación: '{input_string}'")
        print("-" * 40)
        if not trace:
            print("No hubo transiciones válidas (cadena vacía o símbolo inválido)")
        else:
            print(f"Estado inicial: {trace[0][0]}")
            for i, (u, sym, v) in enumerate(trace, start=1):
                print(f"Paso {i}: {u} --({sym})-> {v}")
        print("-" * 40)
        print("✅ ACEPTADA" if accepted else "❌ RECHAZADA")

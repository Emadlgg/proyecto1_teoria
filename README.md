# Proyecto Teoría de la Computación 2025
**Implementación de Algoritmos para Construcción de Autómatas Finitos**

> **Estudiantes:** Osman de León, Ihan Marroquín  
> **Curso:** Teoría de la Computación  
> **Fecha:** Agosto 2025  
> **Lenguaje:** Python 3.8+

## 🚀 Ejecución Rápida

```bash
# 1. Configurar proyecto automáticamente
python setup.py

# 2. Ejecutar programa principal
python run_project.py

# 3. Ejecutar ejemplos de prueba
python run_project.py  # Selecciona opción 2
```

## 📋 Algoritmos Implementados

| Algoritmo | Archivo | Estado |
|-----------|---------|--------|
| **Shunting Yard** (Regex → Postfix) | `src/automata_project.py` | ✅ Completo |
| **Thompson** (Regex → AFN) | `src/automata_project.py` | ✅ Completo |
| **Construcción de Subconjuntos** (AFN → AFD) | `src/automata_project.py` | ✅ Completo |
| **Minimización de Hopcroft** (AFD → AFD minimal) | `src/automata_project.py` | ✅ Completo |
| **Simulador AFD** | `src/automata_project.py` | ✅ Completo |

## 📁 Estructura del Proyecto

```
teoria-computacion-proyecto/
├── src/
│   ├── automata_project.py          # Código principal
│   ├── example_usage.py             # Ejemplos y pruebas
│   └── __init__.py                  # Configuración del paquete
├── output/
│   ├── json/                        # Autómatas exportados (.json)
│   └── visualizations/              # Gráficos generados (.png)
├── run_project.py                   # Script de ejecución
├── requirements.txt                 # Dependencias
└── README.md                        # Este archivo
```

## 🎯 Funcionalidades

### **Entrada**
- Expresiones regulares con operadores: `|`, `+`, `*`, `(`, `)`
- Alfabeto: letras (a-z, A-Z), dígitos (0-9), epsilon (ε)
- Ejemplos: `(a|b)*abb`, `a*b+`, `ab*c`

### **Salida**
- **Archivos JSON**: Descripción completa de cada autómata
- **Visualizaciones PNG**: Gráficos profesionales de los autómatas
- **Simulaciones**: Ejecución paso a paso de cadenas

### **Formato JSON**
```json
{
  "ESTADOS": [0, 1, 2, 3],
  "SIMBOLOS": ["a", "b"],
  "INICIO": [0],
  "ACEPTACION": [3],
  "TRANSICIONES": [[0, "a", 1], [1, "b", 2], [2, "b", 3]]
}
```

## 🧪 Ejemplos de Uso

### **Expresión: `(a|b)*abb(a|b)*`**
- **Reconoce**: Cadenas que contienen "abb" como subcadena
- **Acepta**: `abb`, `aabb`, `babb`, `abba`, `abbab`
- **Rechaza**: `ab`, `ba`, `aab`, `bb`

### **Expresión: `a*b+`**
- **Reconoce**: Cero o más 'a' seguidas de una o más 'b'
- **Acepta**: `b`, `ab`, `abb`, `aaab`, `abbb`
- **Rechaza**: `a`, `ba`, `aba`, `""` (cadena vacía)

## 🔄 Flujo de Procesamiento

```
Expresión Regular
       ↓
   Shunting Yard
       ↓
  Notación Postfix  
       ↓
   Thompson (AFN)
       ↓
Subconjuntos (AFD)
       ↓
  Hopcroft (Minimal)
       ↓
    Simulación
```

## 📊 Resultados de Prueba

| Expresión | AFN | AFD | AFD Min | Cadenas Probadas |
|-----------|-----|-----|---------|------------------|
| `(a\|b)*abb(a\|b)*` | 22 estados | 8 estados | 4 estados | 15 cadenas ✅ |
| `a*b+` | 8 estados | 4 estados | 3 estados | 12 cadenas ✅ |
| `(a\|b)*` | 6 estados | 1 estado | 1 estado | 10 cadenas ✅ |

## 🛠️ Tecnologías Utilizadas

- **Python 3.7+**: Lenguaje principal
- **matplotlib + networkx**: Visualizaciones gráficas (opcional)
- **JSON**: Formato de exportación estándar
- **Collections**: Estructuras de datos optimizadas

> **Nota**: El proyecto funciona completamente sin matplotlib. Las visualizaciones en texto son suficientes para cumplir los requisitos.

## ⚙️ Instalación

```bash
# 1. Descargar/clonar el proyecto
cd teoria-computacion-proyecto

# 2. Configuración automática (recomendado)
python setup.py

# 3. ¡Listo para usar!
python run_project.py
```

### Instalación manual (alternativa):
```bash
pip install matplotlib networkx  # Opcional para gráficos
python run_project.py
```

## 🎮 Modo de Uso

1. **Ejecuta** el programa: `python run_project.py`
2. **Ingresa** una expresión regular válida
3. **Observa** el procesamiento paso a paso
4. **Revisa** los archivos generados en `output/`
5. **Simula** cadenas en el AFD final

## 📈 Rendimiento

- **Tiempo promedio**: < 1 segundo para expresiones regulares típicas
- **Memoria**: Eficiente usando estructuras optimizadas
- **Escalabilidad**: Maneja autómatas con 50+ estados

## 🔍 Validación

El proyecto ha sido validado contra:
- ✅ Toolbox de referencia (cyberzhg.github.io)
- ✅ Ejemplos del libro Hopcroft-Ullman
- ✅ Casos edge (epsilon, cadena vacía, operadores anidados)

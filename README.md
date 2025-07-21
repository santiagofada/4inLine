# 4-en-Línea — Algoritmos para la Toma de Decisiones

> Trabajo Final — **Algoritmos para la Toma de Decisiones**  
> Facultad de Matemática, Astronomía, Física y Computación (FaMAF)

Este proyecto implementa el clásico juego **4-en-Línea** (*Connect-4*) con distintos tipos de agentes inteligentes. Se exploran enfoques de **búsqueda adversarial** y **aprendizaje por refuerzo**, integrados en una interfaz gráfica interactiva desarrollada con **Tkinter**.

---

## Contenidos

1. [Contexto Académico](#contexto-académico)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Cómo Ejecutar](#cómo-ejecutar)
5. [Resultados y Presentación](#resultados-y-presentación)

---

## Contexto Académico

Este proyecto resume gran parte del contenido de la materia, aplicando conceptos clave sobre agentes, decisiones secuenciales y aprendizaje:

| Tema                                   | Implementación                                               |
|----------------------------------------|--------------------------------------------------------------|
| Espacios de estado y acciones          | Clase `Connect4` en [`src/game.py`](./src/game.py)           |
| Búsqueda adversarial (Minimax)         | `MinimaxAgent` con poda α–β y tabla de transposición         |
| Heurísticas de evaluación              | `BasicHeuristicAgent` y `AdvancedHeuristicAgent`             |
| Modelos de Markov (MDP)                | Implícitos en la dinámica de Q-Learning                      |
| Aprendizaje por refuerzo (Q-Learning)  | `QLearningAgent` con entrenamiento vía *self-play*           |

Además, se acompaña con [diapositivas de exposición](./2025_TDJ_Presentation.pdf) con visualizaciones, ejemplos y análisis del rendimiento de los agentes.

---

## Fundamentos Teóricos

### Búsqueda Adversarial: Minimax

El algoritmo Minimax se basa en modelar el juego como un problema de decisión secuencial, determinista y de suma cero:

- **Árbol de juego**: se expande recursivamente hasta cierta profundidad `d`.
- **Función de evaluación**: mide cuán favorable es un estado intermedio (alineaciones, posición central, amenazas).
- **Poda α–β**: evita explorar ramas irrelevantes mejorando la eficiencia.
- **Tabla de transposición**: cachea resultados para estados previamente evaluados.

> El agente resultante es determinista y puede jugar casi perfectamente en niveles intermedios de profundidad.

### Aprendizaje por Refuerzo: Q-Learning

Q-Learning aprende directamente a partir de la experiencia sin necesidad de modelar el entorno (es **model-free**):

```python
Q(s, a) ← Q(s, a) + α · (r + γ · max  [Q(s , a´) − Q(s, a)]) #para todo a´ en el conjunto de estados A
```
donde:

- `α`: tasa de aprendizaje
- `γ`: factor de descuento
- `r`: recompensa

Características específicas:

-**Simetría horizontal**: se reutiliza el conocimiento entre estados reflejados.

-**Estrategias de exploración**: ε-greedy y softmax controlan el balance entre explorar y explotar.

-**Entrenamiento flexible**: contra sí mismo o contra agentes heurísticos.

-**Persistencia**: la tabla Q se guarda y carga desde disco.

>Q-Learning converge teóricamente a la política óptima bajo condiciones ideales de exploración.

## Estructura del Proyecto
```
.
├── src/
│   └── agents/               # Implementaciones de agentes (Minimax, Q-Learning, etc.)
├───trainings/                # Algoritmos de entrenamiento de agentes
├── play_interface.py         # Ejecutar con interfaz
├── play.py                   # Ejecutar por consola
├── Game.py                   # Estructura del juego
└── assets/                   # Q-tables y demas material complementario
    └── 2025_TDJ_Presentation.pdf # Diapositivas para la exposición
```

## Cómo Ejecutar
```bash
# Requiere Python 3.x
pip install -r requirements.txt

# Ejecutar GUI con dos agentes (configurables en play.py)
python play_interface.py
```


## Resultados y Presentación
En el archivo [diapositivas de exposición](./2025_TDJ_Presentation.pdf) se encuentran las visualizaciones clave de cada enfoque:

- Comparación entre Q-Learning y Minimax

- Curvas de aprendizaje

- Evaluación de rendimiento por profundidad

- Limitaciones y desafíos

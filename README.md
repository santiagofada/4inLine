# 4-en-Línea — Algoritmos para la Toma de Decisiones

> Trabajo Final para la materia **Algoritmos para la Toma de Decisiones** – FaMAF

Este repositorio implementa el clásico juego **Connect-4** (4-en-línea) junto con varios agentes de decisión basados en búsqueda, heurísticas y **aprendizaje por refuerzo**. Se puede ejecutar en consola o mediante una interfaz gráfica en **Tkinter**.

---

## Tabla de Contenidos
1. [Contexto Académico](#contexto-académico)
3. [Ejemplos de Uso](#ejemplos-de-uso)
4. [Entrenamiento de Agentes](#entrenamiento-de-agentes)
5. [Estructura del Proyecto](#estructura-del-proyecto)
6. [Algoritmos Implementados](#algoritmos-implementados)
7. [Cómo Contribuir](#cómo-contribuir)
8. [Créditos](#créditos)

---

## Contexto Académico
Este proyecto cubre gran parte del programa de la materia (ver [`Algoritmos para la Toma de Decisiones.pdf`](./Algoritmos%20para%20la%20Toma%20de%20Decisiones.pdf)):

| Tema                                   | Implementación en el proyecto                                    |
|----------------------------------------|------------------------------------------------------------------|
| Representación de espacios de estado   | Clase [Connect4](cci:2://file:///Users/sfada/PycharmProjects/4inLine-master/src/game.py:4:0-47:72) en [src/game.py](cci:7://file:///Users/sfada/PycharmProjects/4inLine-master/src/game.py:0:0-0:0)                                |
| Búsqueda adversarial (Minimax)         | [MinimaxAgent](cci:2://file:///Users/sfada/PycharmProjects/4inLine-master/src/agents/minimaxAgent.py:28:0-172:19) con poda α–β y tabla de transposición             |
| Heurísticas de evaluación              | [BasicHeuristicAgent](cci:2://file:///Users/sfada/PycharmProjects/4inLine-master/src/agents/basicHeuristicAgent.py:4:0-30:54) y [AdvancedHeuristicAgent](cci:2://file:///Users/sfada/PycharmProjects/4inLine-master/src/agents/advancedHeuristicAgent.py:4:0-58:29)                 |
| Modelos de Markov (MDP)                | Implícitos en las transiciones del juego                         |
| Aprendizaje por refuerzo (Q-Learning)  | [QLearningAgent](cci:2://file:///Users/sfada/PycharmProjects/4inLine-master/src/agents/qLearningAgent.py:14:0-109:19) + entrenamiento *self-play*                     |

Las diapositivas [[2025_TDJ_Presentation.pdf](cci:7://file:///Users/sfada/PycharmProjects/4inLine-master/2025_TDJ_Presentation.pdf:0:0-0:0)](./2025_TDJ_Presentation.pdf) se usarán en la exposición para mostrar resultados y análisis.


---

## Fundamentos Teóricos

### Minimax: Búsqueda Adversarial
Minimax modela *4-en-línea* como un **juego de suma-cero**.  MAX (nuestro agente) busca maximizar la utilidad mientras que MIN (el oponente) la minimiza.

* Paso de decisión: se genera el **árbol de juego** hasta una profundidad `d` (`depth` en `MinimaxAgent`).
* **Función heurística**: cuando se alcanza la profundidad límite o un nodo terminal, se evalúa el tablero asignando puntajes a alineaciones de 2-3-4 fichas, priorizando la columna central.
* **Poda α–β** reduce nodos explorados descartando ramas donde `α ≥ β`.
* **Tabla de transposición**: se almacena la mejor acción/valor para tableros ya evaluados, evitando recomputar subárboles repetidos.
* Complejidad sin poda: *O(b^d)* (b≈7).  Con poda y ordenamiento heurístico suele rondar *O(b^{d/2})*.

En esta implementación la política resultante es determinista y *óptima* respecto a la profundidad analizada.

### Q-Learning: Aprendizaje por Refuerzo
Q-Learning es un algoritmo **off-policy, model-free** que aprende la función de acción-valor `Q(s,a)` resolviendo un **Proceso de Decisión de Markov (MDP)** mediante iteración temporal-difusa.

Actualización principal:
```python
Q[s, a] += alpha * (r + gamma * max_a_prime Q[s_prime, a_prime] - Q[s, a])
```
Dónde:
* `alpha (α)` – tasa de aprendizaje.
* `gamma (γ)` – descuento de recompensas futuras.
* `r` – recompensa inmediata (+1 ganar, −1 perder, 0 en curso).

Características de la versión implementada:
* **Serialización normalizada**: El tablero se refleja para compartir valores de estados simétricos, reduciendo el tamaño de la tabla.
* **Exploración**: soporte para `ε`-greedy (explotación/exploración) y **softmax** (probabilidad ∝ e^(Q/τ)).  El parámetro `epsilon` decae exponencialmente (`epsilon_decay`) hasta `epsilon_min`.
* **Entrenamiento**: se realiza vía *self-play* (`train_self_play.py`) o contra heurísticas (`train_q.py`).  Se pueden alcanzar millones de episodios en minutos.
* **Persistencia**: la Q-table se guarda/carga en `assets/q_table_*.pkl`.

Teóricamente, Q-Learning converge a la política óptima si cada par `(s,a)` se visita infinitas veces y `α` satisface la condición de Robbins-Monro (∑α=∞, ∑α²<∞).


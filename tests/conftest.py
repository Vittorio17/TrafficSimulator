"""
Fixture condivise per la suite di test di FluxEngine.

Le fixture costruiscono piccoli grafi NetworkX "a mano" che imitano la
struttura prodotta da OSMnx (MultiDiGraph con nodi che hanno attributi
'x'/'y' e archi con 'length'/'capacity'), così i test non dipendono
da rete o da OpenStreetMap.
"""
import sys
import os
import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def tiny_graph():
    """
    Grafo a 4 nodi disposti su una linea: 1 -> 2 -> 3 -> 4
    con anche un arco diretto 1 -> 4 (scorciatoia), utile per
    testare pathfinding e barriere.

        1 --- 2 --- 3 --- 4
        \\___________________/
                (shortcut, più lunga)
    """
    G = nx.MultiDiGraph()
    coords = {
        1: (41.000, 16.000),
        2: (41.001, 16.001),
        3: (41.002, 16.002),
        4: (41.003, 16.003),
    }
    for node_id, (lat, lon) in coords.items():
        G.add_node(node_id, y=lat, x=lon)

    G.add_edge(1, 2, length=100.0, capacity=2)
    G.add_edge(2, 3, length=100.0, capacity=2)
    G.add_edge(3, 4, length=100.0, capacity=2)
    # Scorciatoia diretta, più lunga della somma dei segmenti intermedi
    G.add_edge(1, 4, length=1000.0, capacity=5)
    return G


@pytest.fixture
def disconnected_graph():
    """Due componenti separate: {1,2} e {3,4}, nessun percorso tra i due gruppi."""
    G = nx.MultiDiGraph()
    for node_id, (lat, lon) in {
        1: (41.0, 16.0), 2: (41.001, 16.001),
        3: (42.0, 17.0), 4: (42.001, 17.001),
    }.items():
        G.add_node(node_id, y=lat, x=lon)
    G.add_edge(1, 2, length=50.0, capacity=3)
    G.add_edge(3, 4, length=50.0, capacity=3)
    return G


@pytest.fixture
def two_agents_node_path_matrix():
    """
    node_path_matrix per 2 agenti che condividono lo stesso arco (1->2)
    nel loro step corrente, utile per testare GetEdgeOccupancy.
    Shape: (n_agenti, n_step)
    """
    node_path_matrix = np.array([
        [1, 2, 3],   # agente 0: sta percorrendo 1->2
        [1, 2, 4],   # agente 1: sta percorrendo 1->2 anch'esso
    ], dtype=np.int64)
    current_step_idx = np.array([0, 0], dtype=np.int32)
    active_mask = np.array([True, True])
    agent_types = np.array([0, 1], dtype=np.int8)  # 0=veicolo, 1=camion
    return node_path_matrix, current_step_idx, active_mask, agent_types

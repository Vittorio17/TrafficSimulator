"""
Test per engine/utils.py — funzioni pure (no rete, no I/O).

Questi sono i test a più alto valore del progetto: la logica di traffico,
emissioni, e Level of Service è interamente concentrata qui e determina
la correttezza dei risultati mostrati nei report finali.
"""
import numpy as np
import networkx as nx
import pytest

from engine.utils import (
    IsGreenLight,
    GetEdgeOccupancy,
    ValidateMovement,
    ApplyBarriers,
    PreProcessing,
    globalTrafficLevel,
    ComputeNetworkLOS,
    ComputeInstantEmissions,
)


# ---------------------------------------------------------------------------
# IsGreenLight
# ---------------------------------------------------------------------------

class TestIsGreenLight:

    def test_even_sum_first_half_cycle_is_green(self):
        # u+v pari -> verde per tick % 30 in [0,15)
        assert IsGreenLight(u=2, v=4, tick=0) is True
        assert IsGreenLight(u=2, v=4, tick=14) is True

    def test_even_sum_second_half_cycle_is_red(self):
        assert IsGreenLight(u=2, v=4, tick=15) is False
        assert IsGreenLight(u=2, v=4, tick=29) is False

    def test_odd_sum_is_complementary_to_even_sum(self):
        # u+v dispari deve essere l'esatto opposto della fase pari, stesso tick
        for tick in [0, 5, 14, 15, 20, 29, 30, 44, 45]:
            even_phase = IsGreenLight(u=2, v=4, tick=tick)
            odd_phase = IsGreenLight(u=3, v=4, tick=tick)
            assert even_phase != odd_phase, f"fase identica al tick={tick}"

    def test_cycle_repeats_every_30_ticks(self):
        assert IsGreenLight(2, 4, tick=1) == IsGreenLight(2, 4, tick=31)
        assert IsGreenLight(2, 4, tick=16) == IsGreenLight(2, 4, tick=46)


# ---------------------------------------------------------------------------
# GetEdgeOccupancy
# ---------------------------------------------------------------------------

class TestGetEdgeOccupancy:

    def test_two_agents_on_same_edge_sum_weights(self, two_agents_node_path_matrix):
        node_path_matrix, current_step_idx, active_mask, agent_types = two_agents_node_path_matrix
        occupancy = GetEdgeOccupancy(node_path_matrix, current_step_idx, active_mask, agent_types)
        # agente 0 (veicolo, peso 1.0) + agente 1 (camion, peso 3.0) sull'arco (1,2)
        assert occupancy[(1, 2)] == pytest.approx(4.0)

    def test_inactive_agents_are_excluded(self, two_agents_node_path_matrix):
        node_path_matrix, current_step_idx, active_mask, agent_types = two_agents_node_path_matrix
        active_mask = np.array([True, False])  # il camion è inattivo
        occupancy = GetEdgeOccupancy(node_path_matrix, current_step_idx, active_mask, agent_types)
        assert occupancy[(1, 2)] == pytest.approx(1.0)

    def test_no_active_agents_returns_empty_dict(self, two_agents_node_path_matrix):
        node_path_matrix, current_step_idx, _, agent_types = two_agents_node_path_matrix
        active_mask = np.array([False, False])
        occupancy = GetEdgeOccupancy(node_path_matrix, current_step_idx, active_mask, agent_types)
        assert occupancy == {}

    def test_pedestrian_weight_is_one(self):
        node_path_matrix = np.array([[1, 2, 3]], dtype=np.int64)
        current_step_idx = np.array([0], dtype=np.int32)
        active_mask = np.array([True])
        agent_types = np.array([2], dtype=np.int8)  # pedone
        occupancy = GetEdgeOccupancy(node_path_matrix, current_step_idx, active_mask, agent_types)
        assert occupancy[(1, 2)] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ValidateMovement
# ---------------------------------------------------------------------------

class TestValidateMovement:

    def test_pedestrians_always_move_ignoring_lights_and_capacity(self, tiny_graph):
        node_path_matrix = np.array([[1, 4, 4]], dtype=np.int64)  # nodo 1 ha degree>2 potenzialmente
        current_step_idx = np.array([0], dtype=np.int32)
        occupancy = {(1, 4): 999}  # occupazione enorme, dovrebbe bloccare un veicolo
        result = ValidateMovement(
            idx=0, node_path_matrix=node_path_matrix, current_step_idx=current_step_idx,
            graph=tiny_graph, occupancy=occupancy, tick=0, agent_type=2,
        )
        assert result is True

    def test_vehicle_blocked_when_u_equals_v(self, tiny_graph):
        node_path_matrix = np.array([[2, 2]], dtype=np.int64)
        current_step_idx = np.array([0], dtype=np.int32)
        result = ValidateMovement(
            idx=0, node_path_matrix=node_path_matrix, current_step_idx=current_step_idx,
            graph=tiny_graph, occupancy={}, tick=0, agent_type=0,
        )
        assert result is False

    def test_vehicle_blocked_by_capacity_when_over_threshold(self, tiny_graph, monkeypatch):
        # Forziamo random.random() a restituire sempre 0.0 (< 0.7) per rendere
        # il test deterministico anziché probabilistico.
        import engine.utils as utils_module
        monkeypatch.setattr(utils_module.random, "random", lambda: 0.0)

        node_path_matrix = np.array([[2, 3]], dtype=np.int64)  # nodo 2 ha degree 2, niente semaforo
        current_step_idx = np.array([0], dtype=np.int32)
        occupancy = {(2, 3): 999}  # capacità arco (2,3) è 2, quindi 999 > capacity
        result = ValidateMovement(
            idx=0, node_path_matrix=node_path_matrix, current_step_idx=current_step_idx,
            graph=tiny_graph, occupancy=occupancy, tick=0, agent_type=0,
        )
        assert result is False

    def test_vehicle_moves_when_under_capacity_and_no_red_light(self, tiny_graph):
        node_path_matrix = np.array([[2, 3]], dtype=np.int64)
        current_step_idx = np.array([0], dtype=np.int32)
        occupancy = {(2, 3): 1}  # ben sotto la capacità (2)
        result = ValidateMovement(
            idx=0, node_path_matrix=node_path_matrix, current_step_idx=current_step_idx,
            graph=tiny_graph, occupancy=occupancy, tick=0, agent_type=0,
        )
        assert result is True


# ---------------------------------------------------------------------------
# ApplyBarriers
# ---------------------------------------------------------------------------

class TestApplyBarriers:

    def test_no_barriers_leaves_graph_unchanged(self, tiny_graph):
        original_weight = tiny_graph.get_edge_data(1, 2, 0).get("weight")
        ApplyBarriers(tiny_graph, barriers=None)
        assert tiny_graph.get_edge_data(1, 2, 0).get("weight") == original_weight

    def test_empty_list_leaves_graph_unchanged(self, tiny_graph):
        ApplyBarriers(tiny_graph, barriers=[])
        assert "weight" not in tiny_graph.get_edge_data(1, 2, 0)

    def test_barrier_near_node_sets_huge_weight_on_outgoing_edges(self, tiny_graph):
        # Nodo 2 è a (41.001, 16.001): piazziamo una barriera esattamente lì.
        # ATTENZIONE: graph.neighbors() su un grafo diretto restituisce solo i
        # successori (archi USCENTI). L'arco entrante 1->2 NON viene quindi
        # toccato: solo 2->3 risulta bloccato. Questo test documenta il
        # comportamento attuale; se l'intento di prodotto è "bloccare tutto il
        # traffico che passa per il nodo", questa è una falla da correggere
        # (andrebbero considerati anche i predecessori, o usare un grafo
        # non orientato per la query di adiacenza).
        ApplyBarriers(tiny_graph, barriers=[(41.001, 16.001)])
        edge_2_3 = tiny_graph.get_edge_data(2, 3, 0)
        assert edge_2_3["weight"] == 999999

    def test_barrier_does_not_block_incoming_edge_known_limitation(self, tiny_graph):
        # Vedi nota nel test sopra: l'arco entrante 1->2 resta INVARIATO.
        # Questo test fallirebbe (nel senso buono) il giorno in cui il bug
        # venisse corretto, segnalando che va aggiornato insieme al fix.
        ApplyBarriers(tiny_graph, barriers=[(41.001, 16.001)])
        edge_1_2 = tiny_graph.get_edge_data(1, 2, 0)
        assert "weight" not in edge_1_2 or edge_1_2["weight"] != 999999

    def test_barrier_does_not_affect_unrelated_edges(self, tiny_graph):
        ApplyBarriers(tiny_graph, barriers=[(41.001, 16.001)])  # vicino al nodo 2
        # L'arco scorciatoia 1->4 non è adiacente al nodo 2
        edge_1_4 = tiny_graph.get_edge_data(1, 4, 0)
        assert edge_1_4.get("weight") != 999999


# ---------------------------------------------------------------------------
# PreProcessing
# ---------------------------------------------------------------------------

class TestPreProcessing:

    def test_assigns_capacity_attribute_to_every_edge(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        for u, v, k in tiny_graph.edges(keys=True):
            assert "capacity" in tiny_graph[u][v][k]

    def test_capacity_is_at_least_one(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        for u, v, k in tiny_graph.edges(keys=True):
            assert tiny_graph[u][v][k]["capacity"] >= 1

    def test_longer_edge_gets_higher_or_equal_capacity(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        cap_short = tiny_graph.get_edge_data(1, 2, 0)["capacity"]  # length=100
        cap_long = tiny_graph.get_edge_data(1, 4, 0)["capacity"]   # length=1000
        assert cap_long >= cap_short

    def test_returns_node_list_and_normalized_weights(self, tiny_graph):
        node_ids, weights = PreProcessing(tiny_graph, timeOfDay="Afternoon")
        assert set(node_ids) == set(tiny_graph.nodes())
        assert weights.sum() == pytest.approx(1.0)
        assert len(weights) == len(node_ids)
        assert np.all(weights > 0)

    def test_morning_boosts_central_nodes_vs_afternoon(self):
        # Costruiamo un grafo con un nodo centrale e nodi periferici a
        # distanza > 0.008 dal centro geografico, così il boost "Morning"
        # (dist < 0.008) si applica solo al nodo centrale e produce un
        # vettore di pesi visibilmente diverso da quello non-Morning.
        G = nx.MultiDiGraph()
        G.add_node(1, y=41.000, x=16.000)   # centrale
        G.add_node(2, y=41.050, x=16.000)   # periferico, ~0.05 di distanza
        G.add_node(3, y=40.950, x=16.000)   # periferico
        G.add_edge(1, 2, length=100.0)
        G.add_edge(2, 3, length=100.0)
        G.add_edge(3, 1, length=100.0)

        _, weights_morning = PreProcessing(G.copy(), timeOfDay="Morning")
        _, weights_other = PreProcessing(G.copy(), timeOfDay="Afternoon")
        assert not np.allclose(weights_morning, weights_other)
        # Il nodo centrale (indice 0, id=1) deve avere un peso relativo
        # maggiore al mattino rispetto al resto della giornata.
        assert weights_morning[0] > weights_other[0]


# ---------------------------------------------------------------------------
# globalTrafficLevel
# ---------------------------------------------------------------------------

class TestGlobalTrafficLevel:

    def test_empty_occupancy_returns_zero(self, tiny_graph):
        assert globalTrafficLevel({}, tiny_graph) == 0.0

    def test_average_load_over_capacity_ratio(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        cap_1_2 = tiny_graph.get_edge_data(1, 2, 0)["capacity"]
        occupancy = {(1, 2): cap_1_2}  # esattamente alla capacità -> rapporto 1.0
        assert globalTrafficLevel(occupancy, tiny_graph) == pytest.approx(1.0)

    def test_missing_edge_in_graph_is_skipped_not_crashed(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        # (99, 100) non esiste nel grafo: get_edge_data restituirà None e
        # la funzione non deve sollevare eccezioni, né contarlo nel risultato.
        occupancy = {(99, 100): 5}
        # len(occupancy) è comunque 1, quindi il risultato sarà 0/1 = 0.0
        assert globalTrafficLevel(occupancy, tiny_graph) == 0.0


# ---------------------------------------------------------------------------
# ComputeNetworkLOS
# ---------------------------------------------------------------------------

class TestComputeNetworkLOS:

    def test_empty_occupancy_returns_zero_counts(self, tiny_graph):
        result = ComputeNetworkLOS({}, tiny_graph)
        assert result == {"A-B": 0, "C-D": 0, "E-F": 0}

    def test_low_vc_ratio_classified_as_a_b(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        capacity = tiny_graph.get_edge_data(1, 2, 0)["capacity"]
        occupancy = {(1, 2): capacity * 0.3}  # 0.3 <= 0.4
        result = ComputeNetworkLOS(occupancy, tiny_graph)
        assert result == {"A-B": 1, "C-D": 0, "E-F": 0}

    def test_mid_vc_ratio_classified_as_c_d(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        capacity = tiny_graph.get_edge_data(1, 2, 0)["capacity"]
        occupancy = {(1, 2): capacity * 0.6}  # 0.4 < 0.6 <= 0.8
        result = ComputeNetworkLOS(occupancy, tiny_graph)
        assert result == {"A-B": 0, "C-D": 1, "E-F": 0}

    def test_high_vc_ratio_classified_as_e_f(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        capacity = tiny_graph.get_edge_data(1, 2, 0)["capacity"]
        occupancy = {(1, 2): capacity * 1.5}  # > 0.8 -> congestionato
        result = ComputeNetworkLOS(occupancy, tiny_graph)
        assert result == {"A-B": 0, "C-D": 0, "E-F": 1}

    def test_boundary_vc_ratio_exactly_0_4_is_a_b(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        capacity = tiny_graph.get_edge_data(1, 2, 0)["capacity"]
        occupancy = {(1, 2): capacity * 0.4}
        result = ComputeNetworkLOS(occupancy, tiny_graph)
        assert result["A-B"] == 1

    def test_multiple_edges_are_aggregated_correctly(self, tiny_graph):
        PreProcessing(tiny_graph, timeOfDay="Afternoon")
        cap_12 = tiny_graph.get_edge_data(1, 2, 0)["capacity"]
        cap_23 = tiny_graph.get_edge_data(2, 3, 0)["capacity"]
        occupancy = {
            (1, 2): cap_12 * 0.1,   # A-B
            (2, 3): cap_23 * 2.0,   # E-F
        }
        result = ComputeNetworkLOS(occupancy, tiny_graph)
        assert result == {"A-B": 1, "C-D": 0, "E-F": 1}


# ---------------------------------------------------------------------------
# ComputeInstantEmissions
# ---------------------------------------------------------------------------

class TestComputeInstantEmissions:

    def test_no_active_agents_returns_all_zero(self):
        active_mask = np.array([False, False])
        agent_types = np.array([0, 1], dtype=np.int8)
        can_move = np.array([False, False])
        result = ComputeInstantEmissions(active_mask, agent_types, can_move)
        assert result == {"veicoli": 0.0, "camion": 0.0, "risparmiata_pedoni": 0.0}

    def test_moving_vehicle_emits_more_than_idle_vehicle(self):
        active_mask = np.array([True])
        agent_types = np.array([0], dtype=np.int8)

        moving = ComputeInstantEmissions(active_mask, agent_types, can_move=np.array([True]))
        idle = ComputeInstantEmissions(active_mask, agent_types, can_move=np.array([False]))

        assert moving["veicoli"] > idle["veicoli"]

    def test_truck_emits_more_than_vehicle_in_same_state(self):
        active_mask = np.array([True, True])
        agent_types = np.array([0, 1], dtype=np.int8)  # veicolo, camion
        can_move = np.array([True, True])
        result = ComputeInstantEmissions(active_mask, agent_types, can_move)
        # un solo veicolo e un solo camion, entrambi in movimento
        assert result["camion"] > result["veicoli"]

    def test_pedestrians_only_contribute_to_saved_co2(self):
        active_mask = np.array([True])
        agent_types = np.array([2], dtype=np.int8)
        can_move = np.array([True])
        result = ComputeInstantEmissions(active_mask, agent_types, can_move)
        assert result["veicoli"] == 0.0
        assert result["camion"] == 0.0
        assert result["risparmiata_pedoni"] > 0.0

    def test_inactive_agents_do_not_contribute(self):
        active_mask = np.array([True, False])
        agent_types = np.array([0, 0], dtype=np.int8)
        can_move = np.array([True, True])
        result_two_active = ComputeInstantEmissions(np.array([True, True]), agent_types, can_move)
        result_one_active = ComputeInstantEmissions(active_mask, agent_types, can_move)
        assert result_one_active["veicoli"] < result_two_active["veicoli"]

    def test_mixed_fleet_sums_independently(self):
        active_mask = np.array([True, True, True])
        agent_types = np.array([0, 1, 2], dtype=np.int8)  # veicolo, camion, pedone
        can_move = np.array([True, True, True])
        result = ComputeInstantEmissions(active_mask, agent_types, can_move)
        assert result["veicoli"] > 0
        assert result["camion"] > 0
        assert result["risparmiata_pedoni"] > 0

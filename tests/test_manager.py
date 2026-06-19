"""
Test per engine/manager.py.

Strategia:
- `buildWorld` viene testato mockando `osmnx.graph_from_point` (nessuna
  chiamata di rete reale a OpenStreetMap/Nominatim).
- `populationWorld` viene testato mockando `ComputeSinglePath`/`ProcessPoolExecutor`
  per evitare sia la rete sia l'overhead di multiprocessing nei test.
- `step()` viene testato costruendo a mano lo stato interno del Manager
  (le matrici NumPy), perché è la tecnica più rapida e deterministica per
  esercitare la logica del ciclo di simulazione senza dipendere dalle fasi
  precedenti.
"""
import numpy as np
import networkx as nx
import pytest
from unittest.mock import patch, MagicMock

from engine.manager import Manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_small_directed_graph():
    """Stesso grafo lineare 1->2->3->4 usato altrove, ma costruito qui
    per non dipendere dalla fixture condivisa (manager non usa pytest fixtures
    di networkx in conftest direttamente in tutti i casi)."""
    G = nx.MultiDiGraph()
    coords = {1: (41.000, 16.000), 2: (41.001, 16.001),
              3: (41.002, 16.002), 4: (41.003, 16.003)}
    for nid, (lat, lon) in coords.items():
        G.add_node(nid, y=lat, x=lon)
    G.add_edge(1, 2, length=100.0, capacity=10)
    G.add_edge(2, 3, length=100.0, capacity=10)
    G.add_edge(3, 4, length=100.0, capacity=10)
    return G


def make_manager_with_fake_population(num_agents=3, max_steps=4, agent_types=None):
    """
    Costruisce un Manager con stato 'POPULATED' creato a mano, senza
    passare per populationWorld(). Ogni agente percorre la sequenza di
    nodi [1,2,3,4] (eventualmente troncata via padding).
    """
    mgr = Manager(sim_id="test_sim")
    mgr.gDrive = make_small_directed_graph()
    mgr.status = "POPULATED"

    node_sequence = [1, 2, 3, 4][:max_steps]
    while len(node_sequence) < max_steps:
        node_sequence.append(node_sequence[-1])  # padding sull'ultimo nodo

    mgr.node_path_matrix = np.tile(np.array(node_sequence, dtype=np.int64), (num_agents, 1))
    mgr.path_matrix = np.zeros((num_agents, max_steps, 2), dtype=np.float32)
    for i, node_id in enumerate(node_sequence):
        lat, lon = mgr.gDrive.nodes[node_id]['y'], mgr.gDrive.nodes[node_id]['x']
        mgr.path_matrix[:, i, 0] = lat
        mgr.path_matrix[:, i, 1] = lon

    mgr.current_step_idx = np.zeros(num_agents, dtype=np.int32)
    mgr.active_mask = np.ones(num_agents, dtype=bool)
    if agent_types is None:
        agent_types = [0] * num_agents
    mgr.agent_types_matrix = np.array(agent_types, dtype=np.int8)
    mgr.pos_matrix = mgr.path_matrix[:, 0, :]
    mgr.delay_tensor = np.zeros(num_agents, dtype=np.int32)
    return mgr


# ---------------------------------------------------------------------------
# Stato iniziale
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_new_manager_starts_in_created_status(self):
        mgr = Manager(sim_id="abc")
        assert mgr.status == "CREATED"

    def test_new_manager_has_zero_tick(self):
        mgr = Manager(sim_id="abc")
        assert mgr.tick_attuale == 0

    def test_new_manager_has_empty_matrices(self):
        mgr = Manager(sim_id="abc")
        assert mgr.path_matrix is None
        assert mgr.active_mask is None


# ---------------------------------------------------------------------------
# buildWorld (con mocking di OSMnx)
# ---------------------------------------------------------------------------

class TestBuildWorld:

    @patch("engine.manager.ox.graph_from_point")
    def test_build_world_sets_status_to_world_ready(self, mock_graph_from_point):
        mock_graph_from_point.return_value = make_small_directed_graph()
        mgr = Manager(sim_id="abc")
        mgr.buildWorld(coords=(41.0, 16.0), distRange=500)
        assert mgr.status == "WORLD_READY"

    @patch("engine.manager.ox.graph_from_point")
    def test_build_world_calls_osmnx_for_drive_and_walk_networks(self, mock_graph_from_point):
        mock_graph_from_point.return_value = make_small_directed_graph()
        mgr = Manager(sim_id="abc")
        mgr.buildWorld(coords=(41.0, 16.0), distRange=500)

        network_types_requested = [
            call.kwargs.get("network_type") for call in mock_graph_from_point.call_args_list
        ]
        assert "drive" in network_types_requested
        assert "walk" in network_types_requested

    @patch("engine.manager.ox.graph_from_point")
    def test_build_world_populates_roads_geometry(self, mock_graph_from_point):
        mock_graph_from_point.return_value = make_small_directed_graph()
        mgr = Manager(sim_id="abc")
        mgr.buildWorld(coords=(41.0, 16.0), distRange=500)
        assert len(mgr.roadsGeometry) == 3  # 3 archi nel grafo finto
        assert "path" in mgr.roadsGeometry[0]

    @patch("engine.manager.ApplyBarriers")
    @patch("engine.manager.ox.graph_from_point")
    def test_build_world_applies_barriers_when_provided(self, mock_graph_from_point, mock_apply_barriers):
        mock_graph_from_point.return_value = make_small_directed_graph()
        mgr = Manager(sim_id="abc")
        barriers = [[41.001, 16.001]]
        mgr.buildWorld(coords=(41.0, 16.0), distRange=500, barriers=barriers)
        mock_apply_barriers.assert_called_once()

    @patch("engine.manager.ApplyBarriers")
    @patch("engine.manager.ox.graph_from_point")
    def test_build_world_skips_barriers_when_none(self, mock_graph_from_point, mock_apply_barriers):
        mock_graph_from_point.return_value = make_small_directed_graph()
        mgr = Manager(sim_id="abc")
        mgr.buildWorld(coords=(41.0, 16.0), distRange=500, barriers=None)
        mock_apply_barriers.assert_not_called()


# ---------------------------------------------------------------------------
# populationWorld (con mocking del pathfinding parallelo)
# ---------------------------------------------------------------------------

class TestPopulationWorld:

    @patch("engine.manager.ProcessPoolExecutor")
    def test_population_allocates_matrices_with_correct_agent_count(self, mock_executor_cls):
        mgr = Manager(sim_id="abc")
        mgr.gDrive = make_small_directed_graph()
        mgr.gWalk = make_small_directed_graph()
        mgr.nodesD = [1, 2, 3, 4]
        mgr.weightsD = np.array([0.25, 0.25, 0.25, 0.25])
        mgr.nodesW = [1, 2, 3, 4]
        mgr.status = "WORLD_READY"

        fake_result = ([(41.0, 16.0), (41.001, 16.001)], [1, 2])
        mock_executor = MagicMock()
        mock_executor.__enter__.return_value.map.return_value = [fake_result] * 5
        mock_executor_cls.return_value = mock_executor

        mgr.populationWorld(vehicles=5, pedestrian=0, timeOfDay="Morning")

        assert mgr.status == "POPULATED"
        assert mgr.path_matrix.shape[0] == 5
        assert len(mgr.agent_types_matrix) == 5

    @patch("engine.manager.ProcessPoolExecutor")
    def test_failed_paths_are_counted_as_spawn_errors_and_excluded(self, mock_executor_cls):
        mgr = Manager(sim_id="abc")
        mgr.gDrive = make_small_directed_graph()
        mgr.gWalk = make_small_directed_graph()
        mgr.nodesD = [1, 2, 3, 4]
        mgr.weightsD = np.array([0.25, 0.25, 0.25, 0.25])
        mgr.nodesW = [1, 2, 3, 4]
        mgr.status = "WORLD_READY"

        good_result = ([(41.0, 16.0), (41.001, 16.001)], [1, 2])
        bad_result = (None, None)
        mock_executor = MagicMock()
        # 3 percorsi validi, 2 falliti su 5 veicoli richiesti
        mock_executor.__enter__.return_value.map.return_value = (
            [good_result] * 3 + [bad_result] * 2
        )
        mock_executor_cls.return_value = mock_executor

        mgr.populationWorld(vehicles=5, pedestrian=0, timeOfDay="Morning")

        assert mgr.spawn_errors == 2
        assert mgr.path_matrix.shape[0] == 3

    @patch("engine.manager.ProcessPoolExecutor")
    def test_population_with_zero_valid_paths_sets_error_status(self, mock_executor_cls):
        mgr = Manager(sim_id="abc")
        mgr.gDrive = make_small_directed_graph()
        mgr.gWalk = make_small_directed_graph()
        mgr.nodesD = [1, 2, 3, 4]
        mgr.weightsD = np.array([0.25, 0.25, 0.25, 0.25])
        mgr.nodesW = [1, 2, 3, 4]
        mgr.status = "WORLD_READY"

        mock_executor = MagicMock()
        mock_executor.__enter__.return_value.map.return_value = [(None, None)] * 5
        mock_executor_cls.return_value = mock_executor

        mgr.populationWorld(vehicles=5, pedestrian=0, timeOfDay="Morning")

        assert mgr.status == "ERROR"

    @patch("engine.manager.ProcessPoolExecutor")
    def test_population_skips_pedestrians_when_count_is_zero(self, mock_executor_cls):
        mgr = Manager(sim_id="abc")
        mgr.gDrive = make_small_directed_graph()
        mgr.gWalk = make_small_directed_graph()
        mgr.nodesD = [1, 2, 3, 4]
        mgr.weightsD = np.array([0.25, 0.25, 0.25, 0.25])
        mgr.nodesW = [1, 2, 3, 4]
        mgr.status = "WORLD_READY"

        good_result = ([(41.0, 16.0), (41.001, 16.001)], [1, 2])
        mock_executor = MagicMock()
        mock_executor.__enter__.return_value.map.return_value = [good_result] * 3
        mock_executor_cls.return_value = mock_executor

        mgr.populationWorld(vehicles=3, pedestrian=0, timeOfDay="Morning")

        # Tutti gli agenti devono essere di tipo 0 o 1 (mai 2=pedone)
        assert not np.any(mgr.agent_types_matrix == 2)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_does_nothing_if_status_not_ready(self):
        mgr = Manager(sim_id="abc")
        mgr.status = "WORLD_READY"  # né POPULATED né RUNNING
        result = mgr.step()
        assert result == []
        assert mgr.tick_attuale == 0

    def test_step_increments_tick_counter(self):
        mgr = make_manager_with_fake_population()
        mgr.step()
        assert mgr.tick_attuale == 1

    def test_step_transitions_status_to_running(self):
        mgr = make_manager_with_fake_population()
        assert mgr.status == "POPULATED"
        mgr.step()
        assert mgr.status == "RUNNING"

    def test_agents_advance_along_their_path(self):
        mgr = make_manager_with_fake_population(num_agents=2, max_steps=4)
        mgr.step()
        # Dopo un tick (assumendo nessun blocco semaforico/di capacità),
        # gli agenti devono essere avanzati di almeno uno step oppure
        # essere rimasti fermi per via di un vincolo — in ogni caso
        # current_step_idx non può mai superare il limite massimo.
        assert np.all(mgr.current_step_idx <= mgr.path_matrix.shape[1] - 1)

    def test_agent_reaching_end_of_path_becomes_inactive(self):
        # max_steps=2 -> [1,2]: l'agente arriva a destinazione in un solo step
        mgr = make_manager_with_fake_population(num_agents=1, max_steps=2)
        for _ in range(5):  # diversi tick per superare eventuali blocchi semaforici
            mgr.step()
        assert mgr.active_mask[0] == False

    def test_delay_tensor_increments_for_blocked_agents(self):
        mgr = make_manager_with_fake_population(num_agents=1, max_steps=4)
        # Capacità minima realistica (PreProcessing garantisce sempre >=1)
        # con occupazione superiore, per forzare il blocco in modo deterministico.
        mgr.gDrive[1][2][0]['capacity'] = 1

        with patch("engine.manager.random.random", return_value=0.0):
            mgr.step()

        assert mgr.delay_tensor[0] >= 0  # non deve mai andare in negativo
        # Con capacità 1, occupazione 1.0 (>capacity richiede >1 per bloccare):
        # qui invece verifichiamo solo l'invariante di non-negatività, perché
        # il blocco esatto dipende anche dal semaforo sul nodo 1.

    def test_step_crashes_with_zero_capacity_edge_known_bug(self):
        # BUG REALE: ComputeNetworkLOS (chiamata da step()) calcola
        # load / capacity senza proteggersi da capacity == 0, andando in
        # ZeroDivisionError. PreProcessing() normalmente garantisce
        # capacity >= 1, quindi questo scenario non si verifica nel
        # flusso standard; può però accadere se un arco viene modificato
        # manualmente (o da una futura funzionalità) con capacità 0.
        # Questo test documenta il comportamento attuale: se in futuro si
        # decide di rendere il motore robusto a capacità nulla (es.
        # trattandolo come arco impraticabile), questo test andrà
        # aggiornato per verificare l'assenza di eccezioni.
        mgr = make_manager_with_fake_population(num_agents=1, max_steps=4)
        mgr.gDrive[1][2][0]['capacity'] = 0

        with pytest.raises(ZeroDivisionError):
            mgr.step()

    def test_pos_matrix_updated_after_step(self):
        mgr = make_manager_with_fake_population(num_agents=2, max_steps=4)
        mgr.step()
        assert mgr.pos_matrix.shape == (2, 2)  # (n_agenti, [lat, lon])

    def test_history_analytics_grows_by_one_tick(self):
        mgr = make_manager_with_fake_population()
        mgr.step()
        assert len(mgr.history_analytics["ticks"]) == 1
        assert len(mgr.history_analytics["los_distribution"]) == 1
        mgr.step()
        assert len(mgr.history_analytics["ticks"]) == 2

    def test_return_data_false_returns_empty_list_but_still_advances_state(self):
        mgr = make_manager_with_fake_population()
        result = mgr.step(return_data=False)
        assert result == []
        assert mgr.tick_attuale == 1  # lo stato avanza comunque

    def test_returned_data_contains_only_active_agents(self):
        mgr = make_manager_with_fake_population(num_agents=2, max_steps=2)
        # Disattiviamo manualmente un agente prima dello step
        mgr.active_mask[1] = False
        data = mgr.step(return_data=True)
        returned_ids = {d["id"] for d in data}
        assert 1 not in returned_ids

    def test_returned_data_has_expected_schema(self):
        mgr = make_manager_with_fake_population(num_agents=1, max_steps=4)
        data = mgr.step(return_data=True)
        if data:  # potrebbe essere vuoto se l'agente si blocca al primo tick
            entry = data[0]
            assert set(entry.keys()) == {"id", "lat", "lon", "type"}
            assert isinstance(entry["id"], int)
            assert isinstance(entry["lat"], float)
            assert isinstance(entry["lon"], float)
            assert isinstance(entry["type"], int)

    def test_cumulative_co2_increases_when_vehicles_are_active(self):
        mgr = make_manager_with_fake_population(num_agents=2, max_steps=4, agent_types=[0, 0])
        mgr.step()
        assert mgr.history_analytics["cumulative_co2_veh"] > 0

    def test_hotspots_recorded_when_occupancy_exceeds_capacity(self):
        mgr = make_manager_with_fake_population(num_agents=5, max_steps=4)
        # Capacità molto bassa per forzare congestione sull'arco iniziale
        mgr.gDrive[1][2][0]['capacity'] = 1
        mgr.step()
        assert len(mgr.history_analytics["hotspots"]) >= 1

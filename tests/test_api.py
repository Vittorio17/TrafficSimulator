"""
Test di integrazione per api/routes.py.

NOTA ARCHITETTURALE IMPORTANTE: il modulo api/routes.py istanzia un singolo
oggetto Manager a livello di modulo (`sim = Manager(sim_id="sim_01")`),
condiviso da TUTTE le richieste HTTP. Questo significa che:
  1. Il server non supporta simulazioni multiple/concorrenti.
  2. Lo stato persiste tra una richiesta e l'altra (e tra i test, se non
     si interviene). In questa suite ogni test sostituisce manualmente
     `routes.sim` con un Manager pulito per restare isolato.

Questi test usano mocking per evitare chiamate reali a OpenStreetMap.
"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import api.routes as routes_module
from main import app
from engine.manager import Manager
from tests.test_manager import make_small_directed_graph


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_sim():
    """Sostituisce l'istanza globale `sim` con un Manager pulito prima di
    ogni test, e la ripristina dopo, per garantire isolamento tra i test."""
    original = routes_module.sim
    routes_module.sim = Manager(sim_id="sim_test")
    yield
    routes_module.sim = original


# ---------------------------------------------------------------------------
# Rotta di base
# ---------------------------------------------------------------------------

class TestRoot:

    def test_root_returns_online_message(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "Traffic Simulator Server is Online"


# ---------------------------------------------------------------------------
# /api/v1/build
# ---------------------------------------------------------------------------

class TestBuildEndpoint:

    @patch("engine.manager.ox.graph_from_point")
    def test_build_with_valid_coordinates_returns_200(self, mock_graph, client):
        mock_graph.return_value = make_small_directed_graph()
        response = client.post("/api/v1/build", json={
            "center": {"lat": 41.0, "lng": 16.0},
            "radius": 500,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "WORLD_READY"

    def test_build_with_invalid_latitude_returns_422(self, client):
        response = client.post("/api/v1/build", json={
            "center": {"lat": 999.0, "lng": 16.0},  # fuori range [-90,90]
            "radius": 500,
        })
        assert response.status_code == 422

    def test_build_with_invalid_longitude_returns_422(self, client):
        response = client.post("/api/v1/build", json={
            "center": {"lat": 41.0, "lng": 999.0},  # fuori range [-180,180]
            "radius": 500,
        })
        assert response.status_code == 422

    def test_build_with_radius_exceeding_max_returns_422(self, client):
        response = client.post("/api/v1/build", json={
            "center": {"lat": 41.0, "lng": 16.0},
            "radius": 999999,  # > maxRadius (20000)
        })
        assert response.status_code == 422

    def test_build_with_negative_radius_returns_422(self, client):
        response = client.post("/api/v1/build", json={
            "center": {"lat": 41.0, "lng": 16.0},
            "radius": -100,
        })
        assert response.status_code == 422

    def test_build_without_radius_uses_default(self, client):
        # radius ha un default (1500): la richiesta deve essere accettata
        # anche senza specificarlo esplicitamente.
        with patch("engine.manager.ox.graph_from_point") as mock_graph:
            mock_graph.return_value = make_small_directed_graph()
            response = client.post("/api/v1/build", json={
                "center": {"lat": 41.0, "lng": 16.0},
            })
        assert response.status_code == 200

    @patch("engine.manager.ox.graph_from_point")
    def test_build_failure_returns_500(self, mock_graph, client):
        mock_graph.side_effect = RuntimeError("OSM unavailable")
        response = client.post("/api/v1/build", json={
            "center": {"lat": 41.0, "lng": 16.0},
            "radius": 500,
        })
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# /api/v1/populate
# ---------------------------------------------------------------------------

class TestPopulateEndpoint:

    def test_populate_before_build_returns_400(self, client):
        response = client.post("/api/v1/populate", json={"vehicle_count": 100})
        assert response.status_code == 400

    @patch("engine.manager.ox.graph_from_point")
    def test_populate_after_build_returns_200_and_starts_background_task(self, mock_graph, client):
        mock_graph.return_value = make_small_directed_graph()
        client.post("/api/v1/build", json={
            "center": {"lat": 41.0, "lng": 16.0}, "radius": 500,
        })
        response = client.post("/api/v1/populate", json={"vehicle_count": 10})
        assert response.status_code == 200
        assert response.json()["status"] == "POPULATING"

    def test_populate_with_negative_vehicle_count_returns_422(self, client):
        response = client.post("/api/v1/populate", json={"vehicle_count": -5})
        assert response.status_code == 422

    def test_populate_with_pedestrian_count_exceeding_max_returns_422(self, client):
        response = client.post("/api/v1/populate", json={
            "vehicle_count": 10,
            "pedestrian_count": 99999999,  # > maxPedestrian (500000)
        })
        assert response.status_code == 422

    def test_populate_missing_required_vehicle_count_returns_422(self, client):
        response = client.post("/api/v1/populate", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/step
# ---------------------------------------------------------------------------

class TestStepEndpoint:

    def test_step_before_population_returns_400(self, client):
        response = client.post("/api/v1/step")
        assert response.status_code == 400

    def test_step_after_population_returns_200_with_expected_schema(self, client):
        # Stato POPULATED costruito a mano per non dipendere da rete/multiprocessing
        routes_module.sim.status = "POPULATED"
        routes_module.sim.gDrive = make_small_directed_graph()
        routes_module.sim.node_path_matrix = np.array([[1, 2, 3]], dtype=np.int64)
        routes_module.sim.path_matrix = np.zeros((1, 3, 2), dtype=np.float32)
        routes_module.sim.current_step_idx = np.zeros(1, dtype=np.int32)
        routes_module.sim.active_mask = np.ones(1, dtype=bool)
        routes_module.sim.agent_types_matrix = np.array([0], dtype=np.int8)
        routes_module.sim.pos_matrix = routes_module.sim.path_matrix[:, 0, :]
        routes_module.sim.delay_tensor = np.zeros(1, dtype=np.int32)

        response = client.post("/api/v1/step")
        assert response.status_code == 200
        body = response.json()
        assert "tick" in body
        assert "status" in body
        assert "agents_count" in body
        assert body["tick"] == 1


# ---------------------------------------------------------------------------
# /api/v1/status
# ---------------------------------------------------------------------------

class TestStatusEndpoint:

    def test_status_on_fresh_simulation(self, client):
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "CREATED"
        assert body["tick"] == 0
        assert body["agents"] == 0


# ---------------------------------------------------------------------------
# /api/v1/analytics
# ---------------------------------------------------------------------------

class TestAnalyticsEndpoint:

    def test_analytics_before_any_simulation_returns_400(self, client):
        response = client.get("/api/v1/analytics")
        assert response.status_code == 400

    def test_analytics_after_build_returns_200_with_defaults(self, client):
        routes_module.sim.status = "WORLD_READY"
        response = client.get("/api/v1/analytics")
        assert response.status_code == 200
        body = response.json()
        assert body["current_los_distribution"] == {"A-B": 0, "C-D": 0, "E-F": 0}
        assert body["top_bottlenecks"] == []

    def test_analytics_reports_top_bottlenecks_sorted_descending(self, client):
        routes_module.sim.status = "RUNNING"
        routes_module.sim.history_analytics["hotspots"] = {
            "Via Roma (1->2)": 3,
            "Via Garibaldi (3->4)": 10,
            "Corso Italia (5->6)": 7,
        }
        response = client.get("/api/v1/analytics")
        body = response.json()
        loads = [h["congested_ticks"] for h in body["top_bottlenecks"]]
        assert loads == sorted(loads, reverse=True)
        assert body["top_bottlenecks"][0]["edge"] == "Via Garibaldi (3->4)"

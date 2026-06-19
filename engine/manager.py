import osmnx as ox
import random
import numpy as np
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from engine.utils import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def computeSinglePath(nodes_pair, graph_data):
    """Worker function for parallel pathfinding computation."""
    start_node, end_node = nodes_pair
    try:
        path = ox.shortest_path(graph_data, start_node, end_node, weight='length')
        if path:
            coords = [(graph_data.nodes[n]['y'], graph_data.nodes[n]['x']) for n in path]
            return coords, path
    except Exception as e:
        logger.error(f"Failed to compute path: {e}")
        return None, None
    return None, None

class Manager:
    
    def __init__(self, sim_id):
        self.id = sim_id
        self.status = "CREATED"
        
        # Vectorized Data Structures
        self.path_matrix = None
        self.node_path_matrix = None
        self.current_step_idx = None
        self.pos_matrix = None
        self.active_mask = None
        self.agent_types_matrix = None
        self.delay_tensor = None
        
        self.history_analytics = {
            "ticks": [],
            "los_distribution": [],  
            "cumulative_co2_veh": 0.0,
            "cumulative_co2_truck": 0.0,
            "cumulative_co2_saved": 0.0,
            "co2_timeline": [],   
            "hotspots": {}           
        }

        self.coords = None
        self.distRange = None
        self.gDrive = None
        self.gWalk = None
        self.roadsGeometry = []
        self.nodesD = None
        self.weightsD = None
        self.nodesW = None

        self.spawn_errors = 0 
        self.tick_count = 0
        self.current_congestion = 0

    def buildWorld(self, coords, dist_range, barriers=None):
        """Initializes the road network and applies physical barriers if provided."""
        self.coords = coords
        self.distRange = dist_range
       
        try:
            self.gDrive = ox.graph_from_point(self.coords, dist=self.distRange, network_type='drive', simplify=True)
            self.gWalk = ox.graph_from_point(self.coords, dist=self.distRange, network_type='walk', simplify=True)
                
            self.nodesD, self.weightsD = PreProcessing(self.gDrive)
            self.nodesW = list(self.gWalk.nodes())
            
            self.roadsGeometry = [{"path": [[round(self.gDrive.nodes[u]['x'], 5), round(self.gDrive.nodes[u]['y'], 5)], 
                            [round(self.gDrive.nodes[v]['x'], 5), round(self.gDrive.nodes[v]['y'], 5)]]} 
                            for u, v in self.gDrive.edges()]

            if barriers: 
                ApplyBarriers(self.gDrive, barriers)
            self.status = "WORLD_READY"
            logger.info("World successfully initialized.")
        except Exception as e:
            self.status = "ERROR"
            logger.error(f"Failed to build world: {e}")

    def populateWorld(self, vehicles, pedestrian, time_of_day):
        """Allocates agent trajectories into vectorized memory structures."""
        all_paths = []
        all_node_paths = []
        agent_types = []
        
        all_paths, all_node_paths, agent_types = self._vehiclesPopulation(vehicles, time_of_day, all_paths, all_node_paths, agent_types)
        all_paths, all_node_paths, agent_types = self._pedestrianPopulation(pedestrian, time_of_day, all_paths, all_node_paths, agent_types)
        
        if not all_paths:
            self.status = "ERROR"
            logger.warning("Population failed: No valid paths generated.")
            return
            
        num_agents = len(all_paths)
        max_steps = max(len(p) for p in all_paths)
        
        # Memory allocation using float32/int64 tensors
        self.path_matrix = np.zeros((num_agents, max_steps, 2), dtype=np.float32)
        self.node_path_matrix = np.zeros((num_agents, max_steps), dtype=np.int64)
        
        for i in range(num_agents):
            path_np = np.array(all_paths[i], dtype=np.float32)
            nodes_np = np.array(all_node_paths[i], dtype=np.int64)
            actual_len = len(path_np)
            
            self.path_matrix[i, :actual_len, :] = path_np
            self.node_path_matrix[i, :actual_len] = nodes_np
            
            if actual_len < max_steps:
                self.path_matrix[i, actual_len:, :] = path_np[-1]
                self.node_path_matrix[i, actual_len:] = nodes_np[-1]
        
        self.current_step_idx = np.zeros(num_agents, dtype=np.int32)
        self.active_mask = np.ones(num_agents, dtype=bool)
        self.agent_types_matrix = np.array(agent_types, dtype=np.int8)
        self.pos_matrix = self.path_matrix[:, 0, :]
        self.delay_tensor = np.zeros(num_agents, dtype=np.int32)        
        self.status = "POPULATED"
        logger.info(f"Population completed: {num_agents} agents ready.")
        
    def _vehiclesPopulation(self, vehicles, time_of_day, all_paths, all_node_paths, agent_types):
        truck_ratio = 0.15 if time_of_day == "Morning" else 0.05
        pairs = [(random.choice(self.nodesD), random.choices(self.nodesD, weights=self.weightsD, k=1)[0]) for _ in range(vehicles)]
        for _ in range(vehicles):
            agent_types.append(1 if random.random() < truck_ratio else 0)
            
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            results = list(executor.map(partial(computeSinglePath, graph_data=self.gDrive), pairs))
            
        for coords, nodes in results:
            if coords:
                all_paths.append(coords); all_node_paths.append(nodes)
            else:
                self.spawn_errors += 1; agent_types.pop()
        return all_paths, all_node_paths, agent_types

    def _pedestrianPopulation(self, pedestrian, time_of_day, all_paths, all_node_paths, agent_types):
        if pedestrian <= 0: return all_paths, all_node_paths, agent_types
        pairs = [(random.choice(self.nodesW), random.choice(self.nodesW)) for _ in range(pedestrian)]
        agent_types.extend([2] * pedestrian)
            
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            results = list(executor.map(partial(computeSinglePath, graph_data=self.gWalk), pairs))
            
        for coords, nodes in results:
            if coords:
                all_paths.append(coords); all_node_paths.append(nodes)
            else:
                self.spawn_errors += 1; agent_types.pop()
        return all_paths, all_node_paths, agent_types
            
    def step(self, return_data=True):
        if self.status not in ["RUNNING", "POPULATED"]: return []

        self.tick_count += 1
        self.status = "RUNNING"
        limit = self.path_matrix.shape[1] - 1

        safe_step_idx = np.minimum(self.current_step_idx, limit - 1)
        u_nodes = self.node_path_matrix[np.arange(len(self.current_step_idx)), safe_step_idx]
        v_nodes = self.node_path_matrix[np.arange(len(self.current_step_idx)), safe_step_idx + 1]
        
        self.active_mask[(u_nodes == v_nodes) | (self.current_step_idx >= limit)] = False
        active_indices = np.where(self.active_mask)[0]

        occupancy = GetEdgeOccupancy(self.node_path_matrix, self.current_step_idx, self.active_mask, self.agent_types_matrix)
        can_move = np.zeros_like(self.active_mask, dtype=bool)

        for idx in active_indices:
            if ValidateMovement(idx, self.node_path_matrix, self.current_step_idx, self.gDrive, occupancy, self.tick_count, self.agent_types_matrix[idx]):
                can_move[idx] = True
        
        self.delay_tensor[self.active_mask & (~can_move)] += 1
        self.current_step_idx[can_move] += 1
        self.pos_matrix = self.path_matrix[np.arange(len(self.current_step_idx)), self.current_step_idx]

        # Analytics & Metrics
        emissions = ComputeInstantEmissions(self.active_mask, self.agent_types_matrix, can_move)
        self.history_analytics["ticks"].append(self.tick_count)
        self.history_analytics["los_distribution"].append(ComputeNetworkLOS(occupancy, self.gDrive))
        self.history_analytics["cumulative_co2_veh"] += emissions["veicoli"]
        self.history_analytics["cumulative_co2_truck"] += emissions["camion"]
        self.history_analytics["cumulative_co2_saved"] += emissions["risparmiata_pedoni"]

        # Hotspot detection
        for (u, v), load in occupancy.items():
            edge_data = self.gDrive.get_edge_data(u, v, 0)
            if edge_data and (load / edge_data.get('capacity', 5)) > 1.0:
                street_name = edge_data.get('name', 'Unnamed Road')
                if isinstance(street_name, list): street_name = " / ".join(street_name)
                edge_id = f"{street_name} ({u}->{v})"
                self.history_analytics["hotspots"][edge_id] = self.history_analytics["hotspots"].get(edge_id, 0) + 1

        return [{'id': int(i), 'lat': float(self.pos_matrix[i][0]), 'lon': float(self.pos_matrix[i][1]), 'type': int(self.agent_types_matrix[i])} 
                for i in np.where(self.active_mask)[0]] if return_data else []
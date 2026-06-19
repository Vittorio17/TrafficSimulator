import osmnx as ox
import os
import networkx as nx
import random
import numpy as np
import logging
from scipy.spatial import KDTree

# Standard logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# OSMnx configuration
ox.settings.timeout = 180
ox.settings.use_cache = True

if not os.path.exists("cache"):
    os.makedirs("cache")

# --- PATHFINDING WORKER ---

def ComputePathWorker(args):
    """Computes the shortest path for an agent using parallel processing."""
    graph, agent_id, origin, target, weight_type, type_str = args
    try:
        if not nx.has_path(graph, origin, target):
            logger.warning(f"No path available for agent {agent_id} from {origin} to {target}")
            return None
            
        path = ox.shortest_path(graph, origin, target, weight=weight_type)
        if path:
            coords = [(round(graph.nodes[n]['y'], 5), round(graph.nodes[n]['x'], 5)) for n in path]
            return {'id': agent_id, 'path': path, 'coords': coords, 'type': type_str}
            
    except Exception as e:
        logger.error(f"Error computing path for agent {agent_id}: {e}")
        return None
    return None

# --- TRAFFIC AND SIGNAL LOGIC ---

def IsGreenLight(u, v, tick):
    """Simulates alternating traffic lights every 30 ticks."""
    cycle = 30
    return (tick % cycle < 15) if (u + v) % 2 == 0 else (tick % cycle >= 15)

def GetEdgeOccupancy(node_path_matrix, current_step_idx, active_mask, agent_types):
    """Calculates the current occupancy density on each road segment."""
    occupancy = {}
    active_indices = np.where(active_mask)[0]
    for idx in active_indices:
        step = current_step_idx[idx]
        u = node_path_matrix[idx, step]
        v = node_path_matrix[idx, step + 1]
        # Heavy vehicles take up more road space
        weight = 3.0 if agent_types[idx] == 1 else 1.0
        occupancy[(u, v)] = occupancy.get((u, v), 0) + weight
    return occupancy

def ValidateMovement(idx, node_path_matrix, current_step_idx, graph, occupancy, tick, agent_type):
    """Validates if an agent can move based on traffic signals and congestion."""
    if agent_type == 2: return True 
    
    step = current_step_idx[idx]
    u = node_path_matrix[idx, step]
    v = node_path_matrix[idx, step + 1]
    if u == v: return False
        
    # 1. Traffic signal check
    if graph.degree(u) > 2 and not IsGreenLight(u, v, tick): 
        return False
    
    # 2. Congestion check
    edge_data = graph.get_edge_data(u, v, 0)
    if edge_data:
        capacity = edge_data.get('capacity', 5)
        if occupancy.get((u, v), 0) > capacity and random.random() < 0.7:
            return False
            
    return True

# --- PRE-PROCESSING AND BARRIERS ---

def ApplyBarriers(graph, barriers):
    """Temporarily increases the weight of edges near specified barrier coordinates."""
    if not barriers: return

    nodes_list = list(graph.nodes(data=True))
    node_coords = np.array([(d['y'], d['x']) for _, d in nodes_list])
    tree = KDTree(node_coords)

    _, indices = tree.query(barriers)
    target_nodes = [nodes_list[idx][0] for idx in indices]

    for target_node in set(target_nodes):
        for neighbor in graph.neighbors(target_node):
            for key in graph[target_node][neighbor]:
                graph[target_node][neighbor][key]['weight'] = 999999

def PreProcessing(graph, time_of_day="Morning"):
    """Calculates road weights and capacities based on geometry."""
    VEHICLE_SPACE = 7.0 
    
    edges_data = graph.edges(data=True)
    lengths = np.array([d.get('length', 1.0) for _, _, d in edges_data])
    capacities = np.maximum(1, (lengths / VEHICLE_SPACE).astype(int))
    
    for i, (u, v, k) in enumerate(graph.edges(keys=True)):
        graph[u][v][k]['capacity'] = capacities[i]

    n_ids = list(graph.nodes())
    coords = np.array([(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in n_ids])
    weights = np.array([graph.degree(n) for n in n_ids], dtype=float) + 1.0
    
    if time_of_day == "Morning":
        center = coords.mean(axis=0)
        dist = np.linalg.norm(coords - center, axis=1)
        weights[dist < 0.008] *= 8.0
        
    return n_ids, weights / weights.sum()

def globalTrafficLevel(occupancy, graph):
    """Calculates the average network stress level."""
    if not occupancy: return 0.0
    
    total_stress = 0
    for (u, v), load in occupancy.items():
        edge_data = graph.get_edge_data(u, v, 0)
        if edge_data:
            capacity = edge_data.get('capacity', 5)
            total_stress += (load / capacity)
    return total_stress / len(occupancy)

# --- ANALYSIS AND METRICS ---

def ComputeNetworkLOS(occupancy, graph):
    """Calculates Level of Service (LOS) based on Volume/Capacity (V/C) ratio."""
    los_counts = {"A-B": 0, "C-D": 0, "E-F": 0}
    if not occupancy:
        return los_counts

    for (u, v), load in occupancy.items():
        edge_data = graph.get_edge_data(u, v, 0)
        if edge_data:
            capacity = edge_data.get('capacity', 5)
            vc_ratio = load / capacity

            if vc_ratio <= 0.4:
                los_counts["A-B"] += 1
            elif vc_ratio <= 0.8:
                los_counts["C-D"] += 1
            else:
                los_counts["E-F"] += 1
                
    return los_counts

def ComputeInstantEmissions(active_mask, agent_types, can_move):
    """Calculates CO2 emissions for the current tick."""
    if not np.any(active_mask):
        return {"veicoli": 0.0, "camion": 0.0, "risparmiata_pedoni": 0.0}

    active_types = agent_types[active_mask]
    active_moving = can_move[active_mask]

    CO2_VEHICLE_MOVING = 2.5
    CO2_VEHICLE_IDLE = 1.2
    CO2_TRUCK_MOVING = 8.5
    CO2_TRUCK_IDLE = 4.5
    CO2_PEDESTRIAN_SAVED = 2.0 

    is_veh = (active_types == 0)
    is_truck = (active_types == 1)
    is_ped = (active_types == 2)

    co2_veh = np.sum(np.where(active_moving & is_veh, CO2_VEHICLE_MOVING, 0.0)) + \
              np.sum(np.where((~active_moving) & is_veh, CO2_VEHICLE_IDLE, 0.0))

    co2_truck = np.sum(np.where(active_moving & is_truck, CO2_TRUCK_MOVING, 0.0)) + \
                np.sum(np.where((~active_moving) & is_truck, CO2_TRUCK_IDLE, 0.0))

    co2_saved = np.sum(np.where(is_ped, CO2_PEDESTRIAN_SAVED, 0.0))

    return {
        "veicoli": round(float(co2_veh), 2),
        "camion": round(float(co2_truck), 2),
        "risparmiata_pedoni": round(float(co2_saved), 2)
    }
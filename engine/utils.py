import osmnx as ox
import os
import networkx as nx
import random
import numpy as np
from scipy.spatial import KDTree

# Configurazione OSMnx
ox.settings.timeout = 180
ox.settings.use_cache = True

if not os.path.exists("cache"):
    os.makedirs("cache")


# --- WORKER PER PATHFINDING ---

def ComputePathWorker(args):
    graph, agentId, origin, target, weightType, typeStr = args
    try:
        if nx.has_path(graph, origin, target):
            path = ox.shortest_path(graph, origin, target, weight=weightType)
            if path:
                coords = [(round(graph.nodes[n]['y'], 5), round(graph.nodes[n]['x'], 5)) for n in path]
                return {'id': agentId, 'path': path, 'coords': coords, 'type': typeStr}
    except:
        return None
    return None

# --- LOGICA DI TRAFFICO E SEMAFORI ---

def IsGreenLight(u, v, tick):
    """Simula semafori alternati ogni 30 tick."""
    cycle = 30
    return (tick % cycle < 15) if (u + v) % 2 == 0 else (tick % cycle >= 15)

def GetEdgeOccupancy(node_path_matrix,current_step_idx,active_mask,agent_types):
    """Calcola quanto spazio occupano gli agenti su ogni strada."""
    occupancy = {}
    active_indices = np.where(active_mask)[0]
    for idx in active_indices:
        step = current_step_idx[idx]
        u = node_path_matrix[idx,step]
        v = node_path_matrix[idx,step+1]
        weight = 3.0 if agent_types[idx]==1 else 1.0
        occupancy[(u, v)] = occupancy.get((u, v), 0) + weight
    return occupancy

def ValidateMovement(idx,node_path_matrix,current_step_idx,graph,occupancy,tick,agent_type):
    """Controlla semafori e congestione prima di muovere l'agente."""
    if agent_type == 2: return True 
    
    step = current_step_idx[idx]
    u = node_path_matrix[idx,step]
    v = node_path_matrix[idx,step+1]
    if u == v: return False
        
    # 1. Controllo Semaforico agli incroci
    if graph.degree(u) > 2 and not IsGreenLight(u, v, tick): 
        return False
    
    # 2. Controllo Congestione (Capacity)
    edgeData = graph.get_edge_data(u, v, 0)
    if edgeData:
        capacity = edgeData.get('capacity', 5)
        if occupancy.get((u, v), 0) > capacity and random.random() < 0.7:
            return False
            
    return True

# --- PRE-PROCESSING E BARRIERE ---

def ApplyBarriers(graph, barriers):
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

def PreProcessing(graph, timeOfDay="Morning"):
    """Calcola pesi e capacità delle strade."""
    VEHICLE_SPACE = 7.0 
    
    #Edges Capacities
    edgesData = graph.edges(data=True) #Return a list of roads with them attribute
    lengths = np.array([d.get('length',1.0) for _, _, d in edgesData]) #Use a list comprhension to estract the only value from the key "lenght"
    capacities = np.maximum(1,(lengths/VEHICLE_SPACE).astype(int)) #Create a numpy array to execute mathematic's operations.
    
    #Update graph (essential for ValidateMovement)
    for i,(u,v,k) in enumerate(graph.edges(keys=True)):
        graph[u][v][k]['capacity'] = capacities[i]

    #Spawn probability
    nIds = list(graph.nodes())
    coords = np.array([(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in nIds])
    weights = np.array([graph.degree(n) for n in nIds], dtype=float) + 1.0
    
    # Directional flow based on timing
    if timeOfDay == "Morning":
        center = coords.mean(axis=0) #Geographic center
        dist = np.linalg.norm(coords - center, axis=1) #Finding distance from the center
        weights[dist < 0.008] *= 8.0 #If a node is near to the center, increase the probability of spawn
        
    return nIds, weights / weights.sum()

def globalTrafficLevel(occupancy,graph):
    if not occupancy: return 0.0
    
    totalStress = 0
    for (u,v), load in occupancy.items():
        edgeData = graph.get_edge_data(u,v,0)
        if edgeData:
            capacity = edgeData.get('capacity',5)
            totalStress += (load/capacity)
    return totalStress/len(occupancy)

# --- LOGICA DI ANALISI E METRICHE AVANZATE ---

def ComputeNetworkLOS(occupancy, graph):
    """
    Calcola il Level of Service (LOS) per ogni arco stradale basandosi 
    sul rapporto Volume/Capacità (V/C) ispirato al manuale HCM.
    """
    los_counts = {"A-B": 0, "C-D": 0, "E-F": 0}
    if not occupancy:
        return los_counts

    for (u, v), load in occupancy.items():
        edge_data = graph.get_edge_data(u, v, 0)
        if edge_data:
            capacity = edge_data.get('capacity', 5)
            vc_ratio = load / capacity

            # Mappatura del livello di servizio standardizzato
            if vc_ratio <= 0.4:
                los_counts["A-B"] += 1  # Flusso libero
            elif vc_ratio <= 0.8:
                los_counts["C-D"] += 1  # Flusso sincronizzato / denso
            else:
                los_counts["E-F"] += 1  # Congestione / Instabilità (Gridlock)
                
    return los_counts

def ComputeInstantEmissions(active_mask, agent_types, can_move):
    """
    Calcola le emissioni istantanee di CO2 (in grammi) per il tick corrente.
    Considera fattori differenziati per tipo di agente e stato di movimento.
    """
    if not np.any(active_mask):
        return {"veicoli": 0.0, "camion": 0.0, "risparmiata_pedoni": 0.0}

    # Estraiamo i tipi degli agenti attualmente attivi
    active_types = agent_types[active_mask]
    active_moving = can_move[active_mask]

    # Coefficienti di emissione stimati per tick (g/tick)
    # Emissioni più alte nelle fasi stop-and-go (attivi ma fermi)
    CO2_VEHICLE_MOVING = 2.5
    CO2_VEHICLE_IDLE = 1.2
    CO2_TRUCK_MOVING = 8.5
    CO2_TRUCK_IDLE = 4.5
    CO2_PEDESTRIAN_SAVED = 2.0 # CO2 risparmiata non usando l'auto

    # Inizializzazione contatori
    co2_veh = 0.0
    co2_truck = 0.0
    co2_saved = 0.0

    # Calcolo vettorializzato tramite maschere per gli agenti attivi
    is_veh = (active_types == 0)
    is_truck = (active_types == 1)
    is_ped = (active_types == 2)

    # Veicoli standard (Tipo 0)
    co2_veh += np.sum(np.where(active_moving & is_veh, CO2_VEHICLE_MOVING, 0.0))
    co2_veh += np.sum(np.where((~active_moving) & is_veh, CO2_VEHICLE_IDLE, 0.0))

    # Mezzi Pesanti (Tipo 1)
    co2_truck += np.sum(np.where(active_moving & is_truck, CO2_TRUCK_MOVING, 0.0))
    co2_truck += np.sum(np.where((~active_moving) & is_truck, CO2_TRUCK_IDLE, 0.0))

    # Pedoni (Tipo 2): calcoliamo il beneficio ecologico sistemico
    co2_saved += np.sum(np.where(is_ped, CO2_PEDESTRIAN_SAVED, 0.0))

    return {
        "veicoli": round(co2_veh, 2),
        "camion": round(co2_truck, 2),
        "risparmiata_pedoni": round(co2_saved, 2)
    }
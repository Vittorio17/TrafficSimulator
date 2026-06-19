import osmnx as ox
import random
from concurrent.futures import ProcessPoolExecutor
from engine.utils import *
import numpy as np
from functools import partial
import os

#This function must be outside the class or static because it must be "serializable" to be sent to the cores
def computeSinglePath(nodes_pair, graph_data):
    start_node, end_node = nodes_pair
    try:
        path = ox.shortest_path(graph_data, start_node, end_node, weight='length')
        if path:
            # CORREZIONE: Usa graph_data (il parametro) e path (la variabile definita sopra)
            coords = [(graph_data.nodes[n]['y'], graph_data.nodes[n]['x']) for n in path]
            return coords, path
    except:
        return None, None
    return None, None

class Manager:
    
    def __init__(self, sim_id):
        self.id = sim_id # Simulation id
        self.status = "CREATED"
        
        #Matrici Vettorializzate
        self.path_matrix = None # Contiene tutti i percorsi di tutti gli agenti
        self.current_step_idx = None # (N_agenti) -> [10,5,0,100,...], indica a che punto del percorso si trova ogni agente
        self.pos_matrix = None #(N_agenti)->[[41.1, 16.8], [41.2, 16.9], ...] coordinate correnti
        self.active_mask = None #(N_agenti,) -> [True, True, False, True, ...], True se l'agente si sta muovendo False se è arrivato a destinazione o si è bloccato
        self.agent_types_matrix = None
        
        self.delay_tensor = None  # Contatore di tick bloccati per singolo agente
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
        self.tick_attuale = 0
        self.current_congestion = 0

    def buildWorld(self, coords, distRange, barriers=None):
        self.coords = coords
        self.distRange = distRange
       
        self.gDrive = ox.graph_from_point(self.coords, dist=self.distRange, network_type='drive', simplify=True)
        self.gWalk = ox.graph_from_point(self.coords, dist=self.distRange, network_type='walk', simplify=True)
            
        self.nodesD, self.weightsD = PreProcessing(self.gDrive)
        self.nodesW = list(self.gWalk.nodes())
        
        self.roadsGeometry = [{"path": [[round(self.gDrive.nodes[u]['x'], 5), round(self.gDrive.nodes[u]['y'], 5)], 
                        [round(self.gDrive.nodes[v]['x'], 5), round(self.gDrive.nodes[v]['y'], 5)]]} 
                        for u,v in self.gDrive.edges()]

        if barriers: ApplyBarriers(self.gDrive, barriers)
        self.status = "WORLD_READY"


    def populationWorld(self, vehicles, pedestrian, timeOfDay):
        all_paths = []
        all_node_paths = []
        agent_types = [] # 0 for vehicles, 1 for pedestrians
        
        all_paths,all_node_paths,agent_types = self.vehiclesPopulation(vehicles,timeOfDay,all_paths,all_node_paths,agent_types)
        all_paths,all_node_paths,agent_types = self.pedestrianPopulation(pedestrian,timeOfDay,all_paths,all_node_paths,agent_types)
        
        if not all_paths:
            self.status = "ERROR"
            return
        #DIMENSIONI MATRICE
        num_agents = len(all_paths)
        max_steps = max(len(p) for p in all_paths)
        #Allocazione della memoria (Tensor 3D), float32 per risparmiare ram senza perdere precisione
        self.path_matrix = np.zeros((num_agents,max_steps,2),dtype=np.float32)
        self.node_path_matrix = np.zeros((num_agents, max_steps), dtype=np.int64) #int64 per ID OpenStreetMap
        #Riempimento delle matrici
        for i in range(num_agents):
            path_np = np.array(all_paths[i], dtype=np.float32)
            nodes_np = np.array(all_node_paths[i], dtype=np.int64)
            actual_len = len(path_np)
            
            # Riempimento
            self.path_matrix[i, :actual_len, :] = path_np
            self.node_path_matrix[i, :actual_len] = nodes_np
            
            # Padding (l'agente resta sull'ultimo nodo/coordinata)
            if actual_len < max_steps:
                self.path_matrix[i, actual_len:, :] = path_np[-1]
                self.node_path_matrix[i, actual_len:] = nodes_np[-1]
        
        #Matrici di stato
        self.current_step_idx = np.zeros(num_agents,dtype=np.int32)
        self.active_mask = np.ones(num_agents, dtype=bool)
        self.agent_types_matrix = np.array(agent_types, dtype=np.int8)
        self.pos_matrix = self.path_matrix[:, 0, :]
        self.delay_tensor = np.zeros(num_agents, dtype=np.int32)        
        self.status = "POPULATED"
        print(f"Population completed: {num_agents} agents ready.")
        
    def vehiclesPopulation(self,vehicles,timeOfDay,all_paths,all_node_paths,agent_types):
        truck_ratio = 0.15 if timeOfDay == "Morning" else 0.05
        print(f"Avvio calcolo parallelo per {vehicles} veicoli...")
        pairs = [] #Source/destinatino pairs
        for _ in range(vehicles):
            o, t = random.choice(self.nodesD), random.choices(self.nodesD, weights=self.weightsD, k=1)[0] #Generazione di un punto di partenza e uno di arrivo
            pairs.append((o,t))
            v_type = 1 if random.random()<truck_ratio else 0 #gestione tipo, 1=Heavy,0=Vehicle
            agent_types.append(v_type)
            
        #Parallelizzazione
        num_workers = min(os.cpu_count(),8)
        worker_func = partial(computeSinglePath, graph_data=self.gDrive) #Partial per fissare il grafo e passare solo le coppie
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_func,pairs)) #Map distribuisce le coppie ai core
        for coords,nodes in results:
            if coords:
                all_paths.append(coords)
                all_node_paths.append(nodes)
            else:
                self.spawn_errors+=1 
                agent_types.pop() #Se un percorso fallisce, bisogna rimuovere il tipo aggiunto inizialmente
        return all_paths,all_node_paths,agent_types

    def pedestrianPopulation(self,pedestrian,timeOfDay,all_paths,all_node_paths,agent_types):
        if pedestrian<=0: return all_paths,all_node_paths,agent_types
        print(f"Avvio calcolo parallelo per {pedestrian} pedoni...")

        pairs = []
        for _ in range(pedestrian):
            o, t = random.choice(self.nodesW), random.choice(self.nodesW)
            pairs.append((o,t))
            agent_types.append(2)
        #Parallelizzazione
        num_workers = min(os.cpu_count(),8)
        worker_func = partial(computeSinglePath,graph_data=self.gWalk)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_func,pairs))
        for coords,nodes in results:
            if coords:
                all_paths.append(coords)
                all_node_paths.append(nodes)
            else:
                self.spawn_errors+=1
                agent_types.pop()
        return all_paths,all_node_paths,agent_types
            
    def step(self,return_data=True):
        if self.status!="RUNNING" and self.status!="POPULATED":
            return []

        self.tick_attuale += 1
        self.status = "RUNNING"
        limit = self.path_matrix.shape[1]-1

        safe_step_idx = np.minimum(self.current_step_idx, limit - 1)
        # Controlliamo simultaneamente per tutti gli agenti se il nodo corrente è uguale a quello successivo
        u_nodes = self.node_path_matrix[np.arange(len(self.current_step_idx)), safe_step_idx]
        v_nodes = self.node_path_matrix[np.arange(len(self.current_step_idx)), safe_step_idx + 1]
        # Disattivazione Agenti
        self.active_mask[(u_nodes == v_nodes) | (self.current_step_idx >= limit)] = False

        active_indices = np.where(self.active_mask)[0]

        occupancy = GetEdgeOccupancy(
            self.node_path_matrix, 
            self.current_step_idx, 
            self.active_mask, 
            self.agent_types_matrix
        )

        
        can_move = np.zeros_like(self.active_mask, dtype=bool)

        for idx in active_indices:
            if ValidateMovement(
                idx, self.node_path_matrix, self.current_step_idx, 
                self.gDrive, occupancy, self.tick_attuale, 
                self.agent_types_matrix[idx]
            ): can_move[idx] = True
        
        is_delayed = self.active_mask & (~can_move)
        self.delay_tensor[is_delayed] += 1

        self.current_step_idx[can_move] += 1 #Incremento indice per gli agenti ancora attivi
        #Estrae tutte le coordinate per ogni agenti in un unica volta
        num_agents = len(self.current_step_idx)
        self.pos_matrix = self.path_matrix[np.arange(num_agents), self.current_step_idx] #np.arange crea una lista di indici

        los_now = ComputeNetworkLOS(occupancy, self.gDrive)
        emissions_now = ComputeInstantEmissions(self.active_mask, self.agent_types_matrix, can_move)

        self.history_analytics["ticks"].append(self.tick_attuale)
        self.history_analytics["los_distribution"].append(los_now)
        self.history_analytics["cumulative_co2_veh"] += emissions_now["veicoli"]
        self.history_analytics["cumulative_co2_truck"] += emissions_now["camion"]
        self.history_analytics["cumulative_co2_saved"] += emissions_now["risparmiata_pedoni"]
        self.history_analytics["co2_timeline"].append(emissions_now)

        # Monitoraggio dei colli di bottiglia (Hotspots permanenti)
        for (u, v), load in occupancy.items():
            edge_data = self.gDrive.get_edge_data(u, v, 0)
            if edge_data and (load / edge_data.get('capacity', 5)) > 1.0:
                # 1. Estrazione del nome della strada dai metadati OSM
                street_name = edge_data.get('name', 'Strada Senza Nome')
                # 2. Gestione del caso in cui OSM restituisca una lista di nomi
                if isinstance(street_name, list):
                    street_name = " / ".join(street_name)
                # 3. Creazione di un identificativo leggibile (Nome Strada + ID Nodi per unicità)
                edge_id = f"{street_name} ({u}->{v})"
                self.history_analytics["hotspots"][edge_id] = self.history_analytics["hotspots"].get(edge_id, 0) + 1

        if not return_data:
            return []

        data_tick = []
        active_indices = np.where(self.active_mask)[0] #Filtro per gli agenti attivi
        for idx in active_indices:
            lat,lon = self.pos_matrix[idx]
            data_tick.append({
                'id':int(idx),
                'lat':float(lat),
                'lon':float(lon),
                'type':int(self.agent_types_matrix[idx])
            })
        return data_tick
    
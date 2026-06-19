from fastapi import APIRouter,HTTPException,BackgroundTasks
from api.schemas import WorldCreate,PopulationCreate
from engine.manager import Manager
import numpy as np

router = APIRouter()
sim = Manager(sim_id="sim_01")

@router.post("/build")
async def build(world: WorldCreate):
    try:
        coords = (world.center.lat,world.center.lng) #Passaggio da oggetto Pydantic in una tupla
        sim.buildWorld(coords=coords, distRange=world.radius, barriers=world.barriers)
        return {"message": "Map loaded", "status": sim.status,"roads_count":len(sim.roadsGeometry)}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@router.post("/populate")
async def populate(world: PopulationCreate, background_tasks: BackgroundTasks):
    if sim.status != "WORLD_READY":
        raise HTTPException(status_code=400, detail="You must first initialize the world using /build")
    
    sim.status = "POPULATING"
    background_tasks.add_task(sim.populationWorld,world.vehicle_count,world.pedestrian_count,world.timeOfDay)
    return {"message":"Population process started", "status":sim.status}

@router.post("/step")
async def SimulationStep(return_data: bool=True):
    if sim.status not in ["POPULATED","RUNNING"]:
        raise HTTPException(status_code=400,detail="Simulation not ready for steps")
    try:
        data = sim.step()
        active_count = int(sim.active_mask.sum()) if sim.active_mask is not None else 0
        return {"tick":sim.tick_attuale,
                "status":sim.status,
                "agents_count": active_count,
                "data": data if return_data else []}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/status")
async def get_status():
    try:
        count = 0
        if sim.agent_types_matrix is not None:
            count = len(sim.agent_types_matrix)
    except:
        count = 0

    return {
        "sim_id": sim.id,
        "status": sim.status,
        "tick": sim.tick_attuale,
        "agents": count
    }

@router.get("/analytics")
async def get_simulation_analytics():
    if sim.status == "CREATED":
        raise HTTPException(status_code=400, detail="Nessuna simulazione inizializzata")
    
    # Calcolo dell'Indice di Ritardo della rete (Network Delay Index)
    # Rapporto tra i tick totali trascorsi e il cammino effettivo completato dagli agenti attivi
    active_indices = np.where(sim.active_mask)[0] if sim.active_mask is not None else []
    
    avg_delay_ticks = 0.0
    if len(active_indices) > 0:
        avg_delay_ticks = float(np.mean(sim.delay_tensor[active_indices]))

    # Identificazione dei 5 colli di bottiglia stradali più severi
    sorted_hotspots = sorted(sim.history_analytics["hotspots"].items(), key=lambda item: item[1], reverse=True)[:5]

    return {
        "sim_id": sim.id,
        "total_ticks": sim.tick_attuale,
        "network_delay_index_ticks": round(avg_delay_ticks, 2),
        "current_los_distribution": sim.history_analytics["los_distribution"][-1] if sim.history_analytics["los_distribution"] else {"A-B": 0, "C-D": 0, "E-F": 0},
        "environmental_impact": {
            "cumulative_co2_kg": round((sim.history_analytics["cumulative_co2_veh"] + sim.history_analytics["cumulative_co2_truck"]) / 1000.0, 3),
            "saved_co2_kg": round(sim.history_analytics["cumulative_co2_saved"] / 1000.0, 3),
            "fleet_breakdown": emissions_timeline_tail(sim.history_analytics["co2_timeline"])
        },
        "top_bottlenecks": [{"edge": h[0], "congested_ticks": h[1]} for h in sorted_hotspots],
        "history_full": sim.history_analytics
    }

def emissions_timeline_tail(timeline):
    if not timeline: return {"veicoli": 0, "camion": 0}
    return timeline[-1]
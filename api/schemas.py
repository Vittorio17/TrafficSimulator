from pydantic import BaseModel,Field
from typing import List, Optional

#Parameters
minLatValue = -90
maxLatValue = 90
minLonValue = -180
maxLonValue = 180
defaultRadius = 1500
maxRadius = 20000
minVehicles = 0
defaultPedestrian=0
minPedestrian=0
maxPedestrian=500000

#Input schemes

class Coordinate(BaseModel):
    # ... significa che il campo è obbligatorio
    # ge = greater or equal
    # le = less or equal
    lat: float = Field(...,ge=minLatValue,le=maxLatValue,description="Latitude in decimal degrees")
    lng: float = Field(..., ge=minLonValue, le=maxLonValue, description="Longitude in decimal degrees")

class WorldCreate(BaseModel):
    # Dati necessari per creare il mondo
    center: Coordinate
    radius: int = Field(default=defaultRadius,gt=0,le=maxRadius) 
    barriers: Optional[List[List[float]]] = None #Ogni sottolista rappresenta i punti di una singola barriera, se l'utente non invia il campo il sistema lo imposta a None

class PopulationCreate(BaseModel):
    vehicle_count: int = Field(...,ge=minVehicles)
    pedestrian_count: int = Field(default=defaultPedestrian,ge=minPedestrian,le=maxPedestrian)
    timeOfDay: str = Field(default="Morning")
# Output schemes

class SimulationStatus(BaseModel):
    sim_id: str
    status: str
    agent_count: int

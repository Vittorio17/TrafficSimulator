from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Traffic Simulator API",
    description="API per la gestione della simulazione di traffico urbano",
    version="1.0.0"
)

#Configurazione CORS per la mappa web, evita che il browser blocca le richieste per sicurezza
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

#Rotte del simulatore
app.include_router(router,prefix="/api/v1",tags=["Simulation"])

#Rotta di controllo
@app.get("/")
async def root():
    return{"message":"Traffic Simulator Server is Online","docs":"/docs"}


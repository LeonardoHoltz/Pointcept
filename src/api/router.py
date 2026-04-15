from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .dbscan import run_dbscan, DBSCANRequest
from .ball_query import run_ballquery, BallQueryRequest
from .optics import run_optics, OPTICSRequest

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/dbscan")
def dbscan_request(req: DBSCANRequest):
    return run_dbscan(req)

@app.post("/ballquery")
def ballquery_request(req: BallQueryRequest):
    return run_ballquery(req)

@app.post("/optics")
def optics_request(req: OPTICSRequest):
    return run_optics(req)
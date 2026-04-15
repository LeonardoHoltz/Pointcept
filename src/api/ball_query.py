from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from typing import List, Optional, Any
from pydantic import BaseModel
import numpy as np

class BallQueryRequest(BaseModel):
    data: List[List[float]]
    radius: float = 0.05


# ------------------------------
# SciPy Ball Query + Graph Clustering
# ------------------------------

def scipy_ball_query_clusters(points, radius):
    N = len(points)

    # SciPy KDTree
    tree = cKDTree(points)

    # matriz esparsa de distância (todas conexões dentro do raio)
    # Retorna matriz (sparse) onde A[i, j] = distância entre i e j se < radius
    dist_matrix = tree.sparse_distance_matrix(
        tree,
        max_distance=radius,
        output_type='coo_matrix'
    )

    # Transformar em grafo não-direcionado (adjacência)
    # Usa 1s porque não precisamos da distância
    row = dist_matrix.row
    col = dist_matrix.col
    data = np.ones(len(row), dtype=np.uint8)

    graph = csr_matrix((data, (row, col)), shape=(N, N))

    # Componentes conectadas = clusters
    n_components, labels = connected_components(
        csgraph=graph,
        directed=False
    )

    clusters = []
    for comp_id in range(n_components):
        members = np.where(labels == comp_id)[0].tolist()
        clusters.append(members)

    return clusters


# ------------------------------
# FastAPI route
# ------------------------------

def run_ballquery(req: BallQueryRequest):

    X = np.array(req.data)

    clusters = scipy_ball_query_clusters(
        X,
        radius=req.radius
    )

    noise = []  # opcional adicionar política de noise

    return {
        "clusters": clusters,
        "noise": noise,
        "params": {
            "radius": req.radius
        }
    }

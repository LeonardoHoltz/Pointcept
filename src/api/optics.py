import numpy as np
from sklearn.cluster import OPTICS
from typing import List, Optional, Any
from pydantic import BaseModel

class OPTICSRequest(BaseModel):
    data: List[List[float]]

    # parâmetros principais do OPTICS
    min_samples: int = 5
    max_eps: float = np.inf
    metric: str = "minkowski"
    p: int = 2
    cluster_method: str = "xi"    # "xi" ou "dbscan"
    eps: Optional[float] = None   # usado apenas se cluster_method="dbscan"
    xi: float = 0.05              # usado se cluster_method="xi"
    min_cluster_size: Optional[float] = None
    algorithm: str = "auto"
    leaf_size: int = 30
    n_jobs: Optional[int] = None


def run_optics(req: OPTICSRequest):

    X = np.array(req.data)

    optics = OPTICS(
        min_samples=req.min_samples,
        max_eps=req.max_eps,
        metric=req.metric,
        p=req.p,
        xi=req.xi,
        min_cluster_size=req.min_cluster_size,
        cluster_method=req.cluster_method,
        eps=req.eps,
        algorithm=req.algorithm,
        leaf_size=req.leaf_size,
        n_jobs=req.n_jobs,
    ).fit(X)

    labels = optics.labels_

    clusters = {}
    noise = []

    for idx, label in enumerate(labels):
        if label == -1:
            noise.append(idx)
        else:
            clusters.setdefault(label, []).append(idx)

    cluster_list = [clusters[k] for k in sorted(clusters.keys())]

    return {
        "clusters": cluster_list,
        "noise": noise,
        "params": req.dict()
    }
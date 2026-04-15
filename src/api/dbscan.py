import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Optional, Any
from pydantic import BaseModel

class DBSCANRequest(BaseModel):
    data: List[List[float]]

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    algorithm: str = "auto"
    leaf_size: int = 30
    p: Optional[int] = None
    n_jobs: Optional[int] = None


def run_dbscan(req: DBSCANRequest):

    X = np.array(req.data)

    db = DBSCAN(
        eps=req.eps,
        min_samples=req.min_samples,
        metric=req.metric,
        algorithm=req.algorithm,
        leaf_size=req.leaf_size,
        p=req.p,
        n_jobs=req.n_jobs,
    ).fit(X)

    labels = db.labels_

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
        "params": {
            "eps": req.eps,
            "min_samples": req.min_samples,
            "metric": req.metric,
            "algorithm": req.algorithm,
            "leaf_size": req.leaf_size,
            "p": req.p,
            "n_jobs": req.n_jobs
        }
    }
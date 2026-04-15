import * as THREE from "three";
import configData from '../config/conf.json' assert { type: 'json' };
import { showToast } from "./toast.js";

export function prepareClustering(geometry, isClusteringMode, selectedLabel) {
    // Highlight the points that will be subjected to clustering
    const labels = geometry.getAttribute("semantic_pred");
    const nPoints = labels.count;
    const colors = geometry.getAttribute("color");

    // Add attribute to store colors before clustering mode
    if (isClusteringMode) {
        const colorsBeforeClustering = new Float32Array(colors.array);
        geometry.setAttribute("colors_before_clustering", new THREE.BufferAttribute(colorsBeforeClustering, 3));
        geometry.attributes.colors_before_clustering.needsUpdate = true;
    }

    if (isClusteringMode) {
        for (let i = 0; i < nPoints; i++) {
            const r = colors.getX(i);
            const g = colors.getY(i);
            const b = colors.getZ(i);
            if (labels.getX(i) === selectedLabel) {
                // Apply a blending between the rgb color and white
                colors.setXYZ(i,
                    0.2 * r + 0.8 * 1.0,
                    0.2 * g + 0.8 * 1.0,
                    0.2 * b + 0.8 * 1.0
                );
            }
            else {
                // Keep the original color but darken it
                colors.setXYZ(i,
                    0.2 * r,
                    0.2 * g,
                    0.2 * b
                );
            }
        }
    }
    else {
        // Restore original colors
        const colorsBeforeClustering = geometry.getAttribute("colors_before_clustering");
        for (let i = 0; i < nPoints; i++) {
            const r = colorsBeforeClustering.getX(i);
            const g = colorsBeforeClustering.getY(i);
            const b = colorsBeforeClustering.getZ(i);
            colors.setXYZ(i, r, g, b);
        }
    }
    colors.needsUpdate = true;
};

export async function applyClustering(geometry, algorithm, selectedLabel, parameters) {
    // Placeholder for clustering implementation
    // This function would implement the clustering algorithm (e.g., DBSCAN)
    // and update the geometry attributes accordingly.
    console.log(`Applying ${algorithm} clustering on label ${selectedLabel}`);

    // collect points belonging to the selected label
    const labels = geometry.getAttribute("semantic_pred");
    const positions = geometry.getAttribute("position");
    const pointsToCluster = [];
    const pointIndices = [];
    for (let i = 0; i < labels.count; i++) {
        if (labels.array[i] === selectedLabel) {
            pointsToCluster.push([positions.getX(i), positions.getY(i), positions.getZ(i)]);
            pointIndices.push(i);
        }
    }

    let clusters;
    let res;
    switch(algorithm) {
        case 'DBSCAN':
            res = await runDBSCAN(pointsToCluster, {
                eps: parameters.dbscan.eps,
                min_samples: parameters.dbscan.min_pts,
                metric: "euclidean",
            });
            clusters = res.clusters;
            break;
        case 'Ball-Query':
            res = await runBallQuery(pointsToCluster, {
                radius: parameters.ball_query.radius,
            });
            clusters = res.clusters;
            break;
    }
    
    console.log(`Found ${clusters.length} clusters for label ${selectedLabel}`);
    // color the points based on cluster assignment
    const colors = geometry.getAttribute("color");
    const labelToColor = new Map();
    clusters.forEach((cluster, clusterIdx) => {
        // assign a random color to the cluster
        const color = new THREE.Color(Math.random(), Math.random(), Math.random());
        labelToColor.set(clusterIdx, color);
        cluster.forEach((pointIdxInCluster) => {
            const globalPointIdx = pointIndices[pointIdxInCluster];
            colors.setXYZ(globalPointIdx,
                color.r,
                color.g,
                color.b
            );
        });
    });
    colors.needsUpdate = true;
    showToast("Clustering finished");
}

/**
 * Runs DBSCAN in backend FastAPI.
 * 
 * @param {Array<Array<number>>} points   - point matrix NxD
 * @param {Object} params                 - DBSCAN parameters
 * @returns {Promise<Object>}             - clusters, noise, params
 */
async function runDBSCAN(points, params = {}) {
  const body = {
    data: points,
    eps: params.eps ?? 0.5,
    min_samples: params.min_samples ?? 5,
    metric: params.metric ?? "euclidean",
    algorithm: params.algorithm ?? "auto",
    leaf_size: params.leaf_size ?? 30,
    p: params.p ?? null,
    n_jobs: params.n_jobs ?? null
  };
  return await requestClusteringAPI('dbscan', body);
}

/**
 * Runs Ball-Query in backend FastAPI.
 * 
 * @param {Array<Array<number>>} points   - point matrix NxD
 * @param {Object} params                 - Ball-Query parameters
 * @returns {Promise<Object>}             - clusters, noise, params
 */
async function runBallQuery(points, params = {}) {
  const body = {
    data: points,
    radius: params.radius ?? 0.5
  };
  return await requestClusteringAPI('ballquery', body);
}

async function requestClusteringAPI(algorithm, body) {
    const response = await fetch(`http://${configData.clustering_api.host}:${configData.clustering_api.port}/${algorithm}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });

    if (!response.ok) {
        throw new Error(`Erro no DBSCAN: ${response.statusText}`);
    }

    return await response.json();
}

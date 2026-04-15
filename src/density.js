
export async function computeDensityRadiusGrid(points, radius = 0.1, divisor = 2, chunkSize = 5000, onProgress = null) {
    const N = points.length / 3;
    const cellSize = radius / divisor;
    const r2 = radius * radius;

    const grid = new Map();
    const getKey = (cx, cy, cz) => `${cx},${cy},${cz}`;

    // --- Build the grid ---
    for (let i = 0; i < N; i++) {
        const x = points[3*i];
        const y = points[3*i + 1];
        const z = points[3*i + 2];

        const cx = Math.floor(x / cellSize);
        const cy = Math.floor(y / cellSize);
        const cz = Math.floor(z / cellSize);

        const key = getKey(cx, cy, cz);
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(i);

        // release UI thread occasionally
        if (i % 5000 === 0) await new Promise(r => setTimeout(r, 0));
    }

    const range = Math.ceil(radius / cellSize);
    const offsets = [];
    for (let o = -range; o <= range; o++) offsets.push(o);

    const densities = new Float32Array(N);

    // --- Compute density in chunks ---
    for (let start = 0; start < N; start += chunkSize) {
        const end = Math.min(start + chunkSize, N);

        for (let i = start; i < end; i++) {
            const x = points[3*i];
            const y = points[3*i + 1];
            const z = points[3*i + 2];

            const cx = Math.floor(x / cellSize);
            const cy = Math.floor(y / cellSize);
            const cz = Math.floor(z / cellSize);

            let count = 0;

            for (const ox of offsets)
                for (const oy of offsets)
                    for (const oz of offsets) {
                        const key = getKey(cx + ox, cy + oy, cz + oz);
                        const bucket = grid.get(key);
                        if (!bucket) continue;

                        for (const j of bucket) {
                            if (j === i) continue;

                            const dx = x - points[3*j];
                            const dy = y - points[3*j + 1];
                            const dz = z - points[3*j + 2];

                            if (dx*dx + dy*dy + dz*dz < r2)
                                count++;
                        }
                    }

            densities[i] = count;
        }

        // release UI thread
        await new Promise(r => setTimeout(r, 0));

        if (onProgress) onProgress(end / N);
    }

    // --- Normalize densities to [0,1] ---
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < N; i++) {
        const v = densities[i];
        if (v < min) min = v;
        if (v > max) max = v;
    }

    const out = new Float32Array(N);
    const rangeVal = max - min || 1;
    for (let i = 0; i < N; i++) {
        out[i] = (densities[i] - min) / rangeVal;
    }

    return out;
}


export function computeDensityKNN(points, k = 10) {
    const N = points.length;
    const density = new Array(N);

    // Prebuild array for convenience
    const P = points.map(p => 
        Array.isArray(p) ? p : [p.x, p.y, p.z]
    );

    for (let i = 0; i < N; i++) {
        const dists = [];

        for (let j = 0; j < N; j++) {
            if (i === j) continue;
            const dx = P[i][0] - P[j][0];
            const dy = P[i][1] - P[j][1];
            const dz = P[i][2] - P[j][2];
            dists.push(dx*dx + dy*dy + dz*dz);
        }

        dists.sort((a, b) => a - b);
        const kthDist = Math.sqrt(dists[k-1]);

        density[i] = 1.0 / (kthDist + 1e-9);
    }

    // Normalize to [0, 1]
    const min = Math.min(...density);
    const max = Math.max(...density);

    return density.map(v => (v - min) / (max - min));
}
import * as THREE from "three";
import { parsePCD } from "./parsePCD.js";
import { applyColorMode } from "./applyColorMode.js";
import { applyOffsetVariation } from "./offsetManipulation.js";
import { applySemanticVisibility } from "./updateLabelVisibility.js";
import { prepareClustering, applyClustering } from "./clustering.js";

// Shaders
import vertexShader from "./shaders/vertex.glsl?raw" with { type: "text" };
import fragmentShader from "./shaders/fragment.glsl?raw" with { type: "text" };

export async function loadPCD(url, options = {}) {
    const {
        colorBy = "rgb",
        colormap = "Category10",
        pointSize = 1,
        offset_pred_slider = 0,
    } = options;

    console.log("Parsing %s", url)
    const geometry = await parsePCD(url);
    console.log("Parsing complete")
    await applyColorMode(geometry, colorBy, colormap);
    console.log("Colors applied")
    
    const material = new THREE.ShaderMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 1.0,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms: {
            uSize: { value: options.pointSize }
        }
    });

    const offset_reference_0 = geometry.getAttribute("position").clone();
    const offset_reference_1 = geometry.getAttribute("pred_centroid").clone();
    
    const cloud = new THREE.Points(geometry, material);
    console.log("Point Cloud object created")

    // Update methods
    cloud.recolor = async (colorByNew, colormapNew) => {
        await applyColorMode(cloud.geometry, colorByNew, colormapNew);
    };

    cloud.changeSize = (newSize) => {
        cloud.material.uniforms.uSize.value = newSize;
    };

    cloud.applyOffset = (offsetPredSlider) => {
        applyOffsetVariation(cloud.geometry.getAttribute("position"), offsetPredSlider, offset_reference_0, offset_reference_1);
    };

    cloud.updateLabelVisibility = (labelsVisibility) => {
        applySemanticVisibility(cloud.geometry, cloud.material, labelsVisibility);
    };

    cloud.prepareClustering = (isClusteringMode, selectedLabel) => {
        prepareClustering(cloud.geometry, isClusteringMode, selectedLabel);
    };

    cloud.applyClustering = async (algorithm, selectedLabel, parameters) => {
        await applyClustering(cloud.geometry, algorithm, selectedLabel, parameters);
    };

    return cloud;
}
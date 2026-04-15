import * as THREE from 'three';

// Set up the scene, camera, and renderer as global variables.
import { camera } from "./cameraControls.js"
let scene, renderer;

export function initRenderingContext() {
    scene = new THREE.Scene();
    renderer = new THREE.WebGLRenderer();

    renderer.setSize( window.innerWidth, window.innerHeight );
    // set background to white
    //renderer.setClearColor(0xffffff, 1);

    // Add the renderer to the HTML document
    document.body.appendChild( renderer.domElement );
}

// Render when camera changes
window.addEventListener('cameraChanged', () => {
  render();
});

// Keep renderer size in sync with the window
window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  render();
});

// Initial render
export function render() {
  renderer.render(scene, camera);
}

export { scene };
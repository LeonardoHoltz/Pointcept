import * as THREE from 'three';

// Camera setup
export const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, -8, 8);
camera.lookAt(new THREE.Vector3(0, 0, 0));
camera.up.set(0, 0, 1);

// Movement state
const moveState = {
  forward: false,
  backward: false,
  left: false,
  right: false,
  up: false,
  down: false,
};

let rotating = false;
let prevMouse = { x: 0, y: 0 };

const sensitivity = 0.0025;
const speedBase = 3.0; // base speed
let rafId = null;
let lastTime = 0;

// Dispatch camera changes
function dispatchCameraChanged() {
  window.dispatchEvent(new CustomEvent('cameraChanged', { detail: { camera } }));
}

function anyMovementActive() {
  return (
    moveState.forward ||
    moveState.backward ||
    moveState.left ||
    moveState.right ||
    moveState.up ||
    moveState.down ||
    rotating
  );
}

function startLoop() {
  if (rafId !== null) return;
  lastTime = performance.now();
  rafId = requestAnimationFrame(loop);
}

function stopLoop() {
  if (rafId === null) return;
  cancelAnimationFrame(rafId);
  rafId = null;
}

function loop(t) {
  const delta = Math.min(0.1, (t - lastTime) / 1000);
  lastTime = t;
  updateCamera(delta);
  if (anyMovementActive()) {
    rafId = requestAnimationFrame(loop);
  } else {
    rafId = null;
  }
}

// Update camera position based on orientation
function updateCamera(delta) {
  const moveSpeed = speedBase;

  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);

  const velocity = new THREE.Vector3();
  if (moveState.forward) velocity.add(forward);
  if (moveState.backward) velocity.sub(forward);
  if (moveState.right) velocity.add(right);
  if (moveState.left) velocity.sub(right);
  if (moveState.up) velocity.add(up);
  if (moveState.down) velocity.sub(up);

  if (velocity.lengthSq() > 0) {
    velocity.normalize().multiplyScalar(moveSpeed * delta);
    camera.position.add(velocity);
    dispatchCameraChanged();
  }
}

// Keyboard handling (Shift = descida)
window.addEventListener('keydown', (e) => {
  let changed = false;
  switch (e.code) {
    case 'KeyW': if (!moveState.forward) { moveState.forward = true; changed = true; } break;
    case 'KeyS': if (!moveState.backward) { moveState.backward = true; changed = true; } break;
    case 'KeyA': if (!moveState.left) { moveState.left = true; changed = true; } break;
    case 'KeyD': if (!moveState.right) { moveState.right = true; changed = true; } break;
    case 'Space': if (!moveState.up) { moveState.up = true; changed = true; } break;
    case 'ShiftLeft':
    case 'ShiftRight': if (!moveState.down) { moveState.down = true; changed = true; } break;
  }
  if (changed) startLoop();
});

window.addEventListener('keyup', (e) => {
  let changed = false;
  switch (e.code) {
    case 'KeyW': if (moveState.forward) { moveState.forward = false; changed = true; } break;
    case 'KeyS': if (moveState.backward) { moveState.backward = false; changed = true; } break;
    case 'KeyA': if (moveState.left) { moveState.left = false; changed = true; } break;
    case 'KeyD': if (moveState.right) { moveState.right = false; changed = true; } break;
    case 'Space': if (moveState.up) { moveState.up = false; changed = true; } break;
    case 'ShiftLeft':
    case 'ShiftRight': if (moveState.down) { moveState.down = false; changed = true; } break;
  }
  if (changed && !anyMovementActive()) stopLoop();
});

// Mouse rotation relative to camera axes (6DoF)
window.addEventListener('contextmenu', (e) => e.preventDefault());
window.addEventListener('mousedown', (e) => {
  if (e.button === 2) {
    rotating = true;
    prevMouse.x = e.clientX;
    prevMouse.y = e.clientY;
    startLoop();
  }
});

window.addEventListener('mouseup', (e) => {
  if (e.button === 2) {
    rotating = false;
    if (!anyMovementActive()) stopLoop();
  }
});

window.addEventListener('mousemove', (e) => {
  if (!rotating) return;

  const dx = (e.clientX - prevMouse.x) * sensitivity;
  const dy = (e.clientY - prevMouse.y) * sensitivity;
  prevMouse.x = e.clientX;
  prevMouse.y = e.clientY;

  // Rotation around camera's local axes
  const qYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -dx);
  const qPitch = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -dy);

  // Apply yaw and pitch in camera's local space
  camera.quaternion.multiply(qYaw);
  camera.quaternion.multiply(qPitch);

  dispatchCameraChanged();
});

// Resize handling
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  dispatchCameraChanged();
});

export function triggerCameraRender() {
  dispatchCameraChanged();
}

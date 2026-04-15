
attribute float visibility;
uniform float uSize;

varying float vVisibility;
varying vec3 vColor;

void main() {
	vVisibility = visibility;
	vColor = color;
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
	gl_PointSize = uSize;
}
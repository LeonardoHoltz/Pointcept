
varying float vVisibility;
varying vec3 vColor;

void main() {
	vec4 finalColor;
	if (vVisibility < 0.5) {
		discard;
	}
	else {
		gl_FragColor = vec4( vColor, 1.0 );
	}
	
}
import * as THREE from "three";
import configData from '../config/conf.json' assert { type: 'json' };

const fields_translation = {
	"confidences": "semantic_pred_confs",
	"semantic_gt": "label",
	"semantic_pred": "semantic_preds",
	"instance_gt": "instance_labels",
	"instance_pred": "instance_pred"
}

const labelsToHide = [0.0, 1.0, 2.0];

/**
 * Faz o parse de um arquivo PCD ASCII.
 * Retorna uma BufferGeometry com todos os atributos possíveis:
 * position, normal, intensity, label, rgb, etc.
 */
export async function parsePCD(url) {
	const response = await fetch(url);
	const text = await response.text();

	const lines = text.split("\n");
	const headerEnd = lines.findIndex(l => l.startsWith("DATA"));
	const header = lines.slice(0, headerEnd);
	const dataLines = lines.slice(headerEnd + 1).filter(l => l.trim().length);

	const fields = header.find(l => l.startsWith("FIELDS")).split(" ").slice(1);
	const fieldIndices = Object.fromEntries(fields.map((f, i) => [f, i]));
	console.log("Detected fields:", fieldIndices);

	const geometry = new THREE.BufferGeometry();
	const positions = [];
	const offsets = [];
	const pred_centroid = [];
	const normals = [];
	const confidences = [];
	const semantic_pred = [];
	const semantic_gt = [];
	const instance_pred = [];
	const instance_gt = [];
	const rgbs = [];
	const labelsToHideVisibility = [];
	let x = 1;
	for (const line of dataLines) {
		const v = line.trim().split(/\s+/).map(Number);

		// positions
		positions.push(v[fieldIndices.x], v[fieldIndices.y], v[fieldIndices.z]);

		// offsets
		offsets.push(v[fieldIndices.offs_pred_x], v[fieldIndices.offs_pred_y], v[fieldIndices.offs_pred_z]);

		// predicted centroids
		pred_centroid.push(
			v[fieldIndices.x] + v[fieldIndices.offs_pred_x],
			v[fieldIndices.y] + v[fieldIndices.offs_pred_y],
			v[fieldIndices.z] + v[fieldIndices.offs_pred_z],
		)

		// Normals
		if ("normal_x" in fieldIndices) {
			normals.push(v[fieldIndices.normal_x], v[fieldIndices.normal_y], v[fieldIndices.normal_z]);
		}

		// Confidences
		{
			const f = fields_translation["confidences"];
			if (f in fieldIndices) confidences.push(v[fieldIndices[f]]);
		}

		// Semantic GT
		{
			const f = fields_translation["semantic_gt"];
			if (f in fieldIndices) semantic_gt.push(v[fieldIndices[f]]);
		}

		// Semantic Predictions
		{
			const f = fields_translation["semantic_pred"];
			if (f in fieldIndices) semantic_pred.push(v[fieldIndices[f]]);
		}

		// Instance GT
		{
			const f = fields_translation["instance_gt"];
			if (f in fieldIndices) instance_gt.push(v[fieldIndices[f]]);
		}

		// Instance Predictions
		{
			const f = fields_translation["instance_pred"];
			if (f in fieldIndices) instance_pred.push(v[fieldIndices[f]]);
		}

		// RGB coded as uint32
		if ("rgb" in fieldIndices) {
			const rgbUint = v[fieldIndices.rgb];
			const r = (rgbUint >> 16) & 0xff;
			const g = (rgbUint >> 8) & 0xff;
			const b = rgbUint & 0xff;
			rgbs.push(r / 255, g / 255, b / 255);
		}
	}

	// Visibility attribute to hide specific points
	if (semantic_pred.length) {
		for (let i = 0; i < semantic_pred.length; i++) {
			if (configData.non_clustered_classes.includes(semantic_pred[i])) {
				labelsToHideVisibility.push(0);
			}
			else
				labelsToHideVisibility.push(1);
		}
	}

	// Uses the geometry to hold the data
	geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
	geometry.setAttribute("offset", new THREE.Float32BufferAttribute(offsets, 3));
	geometry.setAttribute("pred_centroid", new THREE.Float32BufferAttribute(pred_centroid, 3));
	if (normals.length) geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
	if (rgbs.length) geometry.setAttribute("rgb", new THREE.Float32BufferAttribute(rgbs, 3));
	if (confidences.length) geometry.setAttribute("confidences", new THREE.Float32BufferAttribute(confidences, 1));
	if (semantic_gt.length) geometry.setAttribute("semantic_gt", new THREE.Float32BufferAttribute(semantic_gt, 1));
	if (semantic_pred.length) geometry.setAttribute("semantic_pred", new THREE.Float32BufferAttribute(semantic_pred, 1));
	if (instance_gt.length) geometry.setAttribute("instance_gt", new THREE.Float32BufferAttribute(instance_gt, 1));
	if (instance_pred.length) geometry.setAttribute("instance_pred", new THREE.Float32BufferAttribute(instance_pred, 1));
	if (labelsToHideVisibility.length) geometry.setAttribute("labels_to_hide_visibility", new THREE.Float32BufferAttribute(labelsToHideVisibility, 1));

	// Initial visibility of all points is visible (1)
	geometry.setAttribute("visibility", new THREE.Float32BufferAttribute(new Array(positions.length / 3).fill(1), 1));

	// guardamos arrays crus pra futuros recálculos de cor
	geometry.userData.raw = {
		positions,
		rgbs,
		normals,
		confidences,
		semantic_gt,
		semantic_pred,
		instance_gt,
		instance_pred,
	};

	return geometry;
}

export function applyOffsetVariation(
    offset_p_to_be_changed,
    value,
    main_reference_position,
    offset_reference_position
) {
    for (let i = 0; i < offset_reference_position.count; i++) {
        // interpolate original positions and offset
        let x = value * offset_reference_position.getX(i) + (1 - value) * main_reference_position.getX(i);
        let y = value * offset_reference_position.getY(i) + (1 - value) * main_reference_position.getY(i);
        let z = value * offset_reference_position.getZ(i) + (1 - value) * main_reference_position.getZ(i);
        offset_p_to_be_changed.setXYZ(i, x, y, z);
        offset_p_to_be_changed.needsUpdate = true
    }
}
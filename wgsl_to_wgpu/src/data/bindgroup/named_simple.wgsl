struct shared_Camera {
	matrix: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> shared_camera : shared_Camera;

struct shared_Camera {
	matrix: mat4x4<f32>,
}

struct settings_Settings {
	somethings: mat4x4<f32>,
}

@group(0) @binding(0)
var shared_camera: shared_Camera;
@group(0) @binding(1)
var shared_Settings: settings_Settings;

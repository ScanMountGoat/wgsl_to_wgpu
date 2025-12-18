enable wgpu_mesh_shader;

@group(0) @binding(0) var<uniform> a: f32;
@group(0) @binding(1) var<uniform> b: f32;
@group(0) @binding(2) var<uniform> c: f32;
@group(0) @binding(3) var<uniform> d: u32;
@group(0) @binding(4) var<uniform> e: u32;
@group(0) @binding(5) var<uniform> f: u32;
@group(0) @binding(6) var<uniform> g: u32;
@group(0) @binding(7) var<uniform> h: u32;
@group(0) @binding(8) var<uniform> i: u32;
@group(0) @binding(9) var<uniform> j: u32;
@group(0) @binding(10) var<uniform> k: u32;
@group(0) @binding(11) var<uniform> l: f64;

fn inner() -> f32 {
    return d;
}

fn inner_double() -> f64 {
    return l;
}

@vertex
fn vs_main() {
        {
        let x = b;
        let y = f;
    }

    let x = h + i;

    switch e {
        default: {
            let y = e;
            return;
        }
    }
}

@fragment
fn fs_main() {
    let z = e;

    loop {
        let z = c;
    }

    if true {
        let x = h + i;
        let y = g;
    }
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let y = inner();
    let z = f;
    loop {
        let w = g;
        let x = h + i;
    }
}

@compute
@workgroup_size(256)
fn main2() {}

@task
@workgroup_size(1)
fn ts_main() -> @builtin(mesh_task_size) vec3<u32> {
    return vec3(h, i, j);
}

struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
}

var<workgroup> mesh_output: MeshOutput;

@mesh(mesh_output)
@workgroup_size(1)
fn ms_main(@builtin(local_invocation_index) index: u32, @builtin(global_invocation_id) id: vec3<u32>) {
    mesh_output.vertex_count = index + i + k;
}

@group(0) @binding(0)
var<storage, read_write> counter : array<atomic<u32>, 1>;

fn add_one() {
    atomicAdd(&counter[0], 1u);
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    add_one();
}
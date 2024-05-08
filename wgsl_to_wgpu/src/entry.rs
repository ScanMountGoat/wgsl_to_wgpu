use naga::{Function, Module};

pub fn fragment_target_count(module: &Module, f: &Function) -> usize {
    match &f.result {
        Some(r) => match r.binding {
            Some(_) => 1,
            None => {
                // Fragment functions should return a single variable or a struct.
                match &module.types[r.ty].inner {
                    naga::TypeInner::Struct { members, .. } => {
                        members.iter().filter(|m| m.binding.is_some()).count()
                    }
                    _ => 0,
                }
            }
        },
        None => 0,
    }
}

#[cfg(test)]
mod test {
    use super::*;
}

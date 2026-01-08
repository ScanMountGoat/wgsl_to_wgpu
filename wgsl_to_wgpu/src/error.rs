use std::path::Path;
use thiserror::Error;

/// Errors while generating Rust source for a WGSL shader module.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CreateModuleError {
    /// Bind group sets must be consecutive and start from 0.
    /// See `bind_group_layouts` for
    /// [PipelineLayoutDescriptor](https://docs.rs/wgpu/latest/wgpu/struct.PipelineLayoutDescriptor.html#).
    #[error("bind groups are non-consecutive or do not start from 0")]
    NonConsecutiveBindGroups,

    /// Each binding resource must be associated with exactly one binding index.
    #[error("duplicate binding found with index `{binding}`")]
    DuplicateBinding { binding: u32 },

    /// The shader source could not be parsed.
    #[error("failed to parse: {error}")]
    ParseError {
        error: naga::front::wgsl::ParseError,
    },

    /// The shader source could not be validated.
    #[error("failed to validate: {error}")]
    ValidationError {
        error: naga::WithSpan<naga::valid::ValidationError>,
    },
}

impl CreateModuleError {
    /// Writes a diagnostic error to stderr.
    pub fn emit_to_stderr(&self, wgsl_source: &str) {
        match self {
            CreateModuleError::ParseError { error } => error.emit_to_stderr(wgsl_source),
            CreateModuleError::ValidationError { error } => error.emit_to_stderr(wgsl_source),
            other => {
                eprintln!("{other}")
            }
        }
    }

    /// Writes a diagnostic error to stderr, including a source path.
    pub fn emit_to_stderr_with_path(&self, wgsl_source: &str, path: impl AsRef<Path>) {
        let path = path.as_ref();
        match self {
            CreateModuleError::ParseError { error } => {
                error.emit_to_stderr_with_path(wgsl_source, path)
            }
            CreateModuleError::ValidationError { error } => {
                let path = path.to_string_lossy();
                error.emit_to_stderr_with_path(wgsl_source, &path)
            }
            other => {
                eprintln!("{}: {}", path.to_string_lossy(), other)
            }
        }
    }

    /// Creates a diagnostic string from the error.
    pub fn emit_to_string(&self, wgsl_source: &str) -> String {
        match self {
            CreateModuleError::ParseError { error } => error.emit_to_string(wgsl_source),
            CreateModuleError::ValidationError { error } => error.emit_to_string(wgsl_source),
            other => {
                format!("{other}")
            }
        }
    }

    /// Creates a diagnostic string from the error, including a source path.
    pub fn emit_to_string_with_path(&self, wgsl_source: &str, path: impl AsRef<Path>) -> String {
        let path = path.as_ref();
        match self {
            CreateModuleError::ParseError { error } => {
                error.emit_to_string_with_path(wgsl_source, path)
            }
            CreateModuleError::ValidationError { error } => {
                let path = path.to_string_lossy();
                error.emit_to_string_with_path(wgsl_source, &path)
            }
            other => {
                format!("{}: {}", path.to_string_lossy(), other)
            }
        }
    }
}

//! DXO Studio native backend: inspect run store reader + loopback HTTP API.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod server;
pub mod store;

pub use server::{InspectApiOptions, InspectApiServer};
pub use store::{RunMetaV0, RunSummary, default_runs_root, list_runs, read_events, read_run_meta};

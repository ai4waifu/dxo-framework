//! DXO Studio desktop GUI: Tauri + embedded VMZ WebUI + native inspect API.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use dxo_studio::{InspectApiOptions, InspectApiServer};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

struct StudioState {
    api_url: String,
    server: Mutex<Option<InspectApiServer>>,
}

#[tauri::command]
fn studio_api_url(state: tauri::State<'_, StudioState>) -> String {
    state.api_url.clone()
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let mut options = InspectApiOptions::default();
            options.port = 0;
            let server = InspectApiServer::bind(options).map_err(|err| err.to_string())?;
            let api_url = server.url.clone();
            let init = format!("window.__DXO_STUDIO_API__={api_url:?};");

            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("DXO Studio")
                .inner_size(1200.0, 800.0)
                .initialization_script(&init)
                .build()
                .map_err(|err| err.to_string())?;

            app.manage(StudioState { api_url, server: Mutex::new(Some(server)) });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![studio_api_url])
        .run(tauri::generate_context!())
        .expect("error while running DXO Studio");
}

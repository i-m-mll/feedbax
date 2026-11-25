use crate::studio_menu::require_project_window;
use tauri::{ipc::InvokeBody, ipc::Request, WebviewWindow};

/// Saves the bytes a project window hands over where the user chooses in the system save panel. Answers whether the
/// file was written: `false` when the user cancels.
#[tauri::command]
pub async fn save_studio_file(window: WebviewWindow, request: Request<'_>) -> Result<bool, String> {
    require_project_window(&window)?;
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err("A saved file is sent as raw bytes".into());
    };
    let name = request
        .headers()
        .get("Studio-File-Name")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| percent_decode(value))
        .ok_or("A saved file needs its name")?;
    choose_and_write(&window, name, bytes.clone()).await
}

fn percent_decode(text: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(text.len());
    let mut rest = text.bytes();
    while let Some(byte) = rest.next() {
        if byte == b'%' {
            let high = (rest.next()? as char).to_digit(16)?;
            let low = (rest.next()? as char).to_digit(16)?;
            bytes.push((high * 16 + low) as u8);
        } else {
            bytes.push(byte);
        }
    }
    String::from_utf8(bytes).ok()
}

#[cfg(target_os = "macos")]
async fn choose_and_write(window: &WebviewWindow, name: String, bytes: Vec<u8>) -> Result<bool, String> {
    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSModalResponse, NSModalResponseOK, NSSavePanel};
    use objc2_foundation::NSString;
    use std::sync::mpsc;

    let (sender, receiver) = mpsc::channel::<Result<bool, String>>();
    window
        .run_on_main_thread(move || {
            let Some(main) = MainThreadMarker::new() else {
                let _ = sender.send(Err("Saving requires the main thread".into()));
                return;
            };
            let panel = NSSavePanel::savePanel(main);
            panel.setNameFieldStringValue(&NSString::from_str(&name));
            panel.setCanCreateDirectories(true);
            let picker = panel.clone();
            let completion = RcBlock::new(move |response: NSModalResponse| {
                picker.orderOut(None);
                let result = if response != NSModalResponseOK {
                    Ok(false)
                } else if let Some(path) = picker.URL().and_then(|url| url.path()) {
                    std::fs::write(path.to_string(), &bytes)
                        .map(|()| true)
                        .map_err(|error| format!("The file could not be written: {error}"))
                } else {
                    Err("The chosen place has no local path".into())
                };
                let _ = sender.send(result);
            });
            panel.beginWithCompletionHandler(&completion);
        })
        .map_err(|error| error.to_string())?;
    tauri::async_runtime::spawn_blocking(move || receiver.recv())
        .await
        .map_err(|error| error.to_string())?
        .map_err(|_| "The save panel closed without an answer".to_string())?
}

#[cfg(not(target_os = "macos"))]
async fn choose_and_write(_window: &WebviewWindow, _name: String, _bytes: Vec<u8>) -> Result<bool, String> {
    Err("Saving a file requires macOS".into())
}

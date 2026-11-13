use crate::{rename_recent, window_session::Folder};
use objc2::MainThreadMarker;
use objc2_app_kit::NSWindow;
use objc2_foundation::{NSString, NSURL};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    sync::{mpsc, Mutex},
    thread,
    time::Duration,
};
use tauri::{AppHandle, Manager, WebviewWindow};

const INTERVAL: Duration = Duration::from_millis(750);
const ROUTE: &str = "http://localhost/_desktop/represented-project";

#[derive(Clone)]
struct Target {
    project: String,
    folder: Folder,
    generation: String,
}

struct Binding {
    target: Target,
    request: u64,
    declaration: Option<PathBuf>,
}

#[derive(Default)]
struct Documents {
    windows: BTreeMap<String, Binding>,
    sequence: u64,
}

impl Documents {
    fn next(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    fn bind(&mut self, label: &str, target: Target, path: PathBuf) {
        let request = self.next();
        self.windows.insert(
            label.into(),
            Binding {
                target,
                request,
                declaration: Some(path),
            },
        );
    }

    fn invalidate(&mut self, label: &str) {
        let request = self.next();
        if let Some(binding) = self.windows.get_mut(label) {
            binding.request = request;
        }
    }

    fn request(&mut self, label: &str) -> Option<(u64, Target)> {
        self.invalidate(label);
        self.windows
            .get(label)
            .map(|binding| (binding.request, binding.target.clone()))
    }

    fn published(&mut self, label: &str, declaration: Option<PathBuf>) {
        if let Some(binding) = self.windows.get_mut(label) {
            binding.declaration = declaration;
        }
    }

    fn changed(&self, label: &str, request: u64, declaration: Option<&Path>) -> Option<&Target> {
        let binding = self.windows.get(label)?;
        if request != binding.request || declaration == binding.declaration.as_deref() {
            return None;
        }
        Some(&binding.target)
    }
}

pub struct ProjectDocuments {
    documents: Mutex<Documents>,
    wake: mpsc::SyncSender<()>,
}

impl ProjectDocuments {
    pub fn start(app: &AppHandle, control: PathBuf) -> tauri::Result<()> {
        if app.try_state::<Self>().is_some() {
            return Ok(());
        }
        let (wake, requests) = mpsc::sync_channel(1);
        let client = Client::builder()
            .unix_socket(control.join("service.sock"))
            .redirect(reqwest::redirect::Policy::none())
            .timeout(INTERVAL)
            .build()
            .map_err(io::Error::other)?;
        if app.manage(Self {
            documents: Mutex::default(),
            wake,
        }) {
            let app = app.clone();
            thread::spawn(move || {
                while !matches!(
                    requests.recv_timeout(INTERVAL),
                    Err(mpsc::RecvTimeoutError::Disconnected)
                ) {
                    if observe(&app, &client).is_err() {
                        break;
                    }
                }
            });
        }
        Ok(())
    }

    pub fn bind(
        window: &WebviewWindow,
        project: &str,
        folder: Folder,
        generation: String,
        path: PathBuf,
    ) -> tauri::Result<()> {
        let target = Target {
            project: project.into(),
            folder,
            generation,
        };
        if !belongs_to(&path, &target.folder) {
            return Err(io::Error::other("Invalid represented Studio declaration").into());
        }
        set_represented(window, Some(&path))?;
        let state = window.state::<Self>();
        state
            .documents
            .lock()
            .unwrap()
            .bind(window.label(), target, path);
        let _ = state.wake.try_send(());
        Ok(())
    }

    pub fn refresh(window: &WebviewWindow) {
        let state = window.state::<Self>();
        state.documents.lock().unwrap().invalidate(window.label());
        let _ = state.wake.try_send(());
    }

    pub fn detach(app: &AppHandle, label: &str) {
        if let Some(state) = app.try_state::<Self>() {
            state.documents.lock().unwrap().windows.remove(label);
        }
    }
}

fn belongs_to(path: &Path, folder: &Folder) -> bool {
    path.parent() == Some(folder.path.as_path())
        && path.extension().is_some_and(|ext| ext == "feedbax")
}

fn resolve(client: &Client, target: &Target) -> Option<PathBuf> {
    #[derive(Deserialize)]
    struct Reply {
        declaration: Option<PathBuf>,
    }
    client.post(ROUTE)
        .json(&json!({
            "project_id": target.project, "folder_identity": target.folder.identity, "generation": target.generation,
        }))
        .send().ok()?.error_for_status().ok()?.json::<Reply>().ok()?
        .declaration.filter(|path| belongs_to(path, &target.folder))
}

fn observe(app: &AppHandle, client: &Client) -> tauri::Result<()> {
    let state = app.state::<ProjectDocuments>();
    let labels: Vec<_> = state
        .documents
        .lock()
        .unwrap()
        .windows
        .keys()
        .cloned()
        .collect();
    for label in labels {
        let Some((request, target)) = state.documents.lock().unwrap().request(&label) else {
            continue;
        };
        let declaration = resolve(client, &target);
        let target_app = app.clone();
        app.run_on_main_thread(move || {
            if let Err(error) = publish(&target_app, &label, request, declaration) {
                eprintln!("Could not update Studio represented document: {error}");
            }
        })?;
    }
    Ok(())
}

fn publish(
    app: &AppHandle,
    label: &str,
    request: u64,
    declaration: Option<PathBuf>,
) -> tauri::Result<()> {
    let Some(window) = app.get_webview_window(label) else {
        return Ok(());
    };
    let state = app.state::<ProjectDocuments>();
    let project = {
        let documents = state.documents.lock().unwrap();
        let Some(target) = documents.changed(label, request, declaration.as_deref()) else {
            return Ok(());
        };
        target.project.clone()
    };
    set_represented(&window, declaration.as_deref())?;
    if let Some(title) = declaration
        .as_deref()
        .and_then(Path::file_stem)
        .and_then(|stem| stem.to_str())
    {
        window.set_title(title)?;
        rename_recent(app, &project, title)?;
    }
    state
        .documents
        .lock()
        .unwrap()
        .published(label, declaration);
    Ok(())
}

fn set_represented(window: &WebviewWindow, path: Option<&Path>) -> tauri::Result<()> {
    MainThreadMarker::new()
        .ok_or_else(|| io::Error::other("Studio document identity requires the main thread"))?;
    let url = path
        .map(|path| {
            path.to_str()
                .map(|path| NSURL::fileURLWithPath_isDirectory(&NSString::from_str(path), false))
                .ok_or_else(|| io::Error::other("Studio declaration path must be valid Unicode"))
        })
        .transpose()?;
    let native = window.ns_window()?.cast::<NSWindow>();
    let native = unsafe { native.as_ref() }
        .ok_or_else(|| io::Error::other("Missing Studio native window"))?;
    native.setRepresentedURL(url.as_deref());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_rejects_obsolete_results_across_invalidation_and_reopen() {
        let target = Target {
            project: "reach".into(),
            folder: Folder {
                path: "/projects/reach".into(),
                identity: [1, 2],
            },
            generation: "service".into(),
        };
        let mut documents = Documents::default();
        let label = "project-reach";
        let original = target.folder.path.join("original.feedbax");
        let renamed = target.folder.path.join("renamed.feedbax");
        documents.bind(label, target.clone(), original.clone());
        let (old, _) = documents.request(label).unwrap();
        let (new, _) = documents.request(label).unwrap();
        assert!(documents.changed(label, new, Some(&renamed)).is_some());
        assert!(documents.changed(label, old, None).is_none());
        documents.published(label, Some(renamed.clone()));
        let (invalid, _) = documents.request(label).unwrap();
        assert!(documents.changed(label, invalid, None).is_some());
        documents.published(label, None);
        let (recovered, _) = documents.request(label).unwrap();
        assert!(documents
            .changed(label, recovered, Some(&renamed))
            .is_some());
        documents.windows.remove(label);
        assert!(documents
            .changed(label, recovered, Some(&renamed))
            .is_none());
        documents.bind(label, target, original);
        assert!(documents
            .changed(label, recovered, Some(&renamed))
            .is_none());
    }
}

#[cfg(target_os = "macos")]
mod native_frame;
mod service;
mod studio_menu;

use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    io,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};
use tauri::{
    menu::{MenuItem, PredefinedMenuItem, Submenu},
    webview::NewWindowResponse,
    App, AppHandle, Manager, RunEvent, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder,
};

static BUSY: AtomicBool = AtomicBool::new(false);
const TITLE: &str = "Feedbax Studio";
const PROJECT_PREFIX: &str = "project-";
const CONNECT_PATH: &str = "/_desktop/connect";

#[derive(Default)]
struct Windows {
    projects_root: Option<PathBuf>,
    status: Option<(String, String)>,
}

#[derive(Deserialize)]
struct OpenedProject {
    id: String,
    title: String,
}

#[derive(Deserialize)]
struct StudioTicket {
    url: Url,
    generation: String,
    project: OpenedProject,
}

impl StudioTicket {
    fn label(&self) -> String {
        format!("{PROJECT_PREFIX}{}", self.project.id)
    }
}

#[derive(Default)]
enum Entry {
    #[default]
    NewProject,
    Folder(PathBuf),
    Reconnect(String),
}

fn status_window(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    if let Some(window) = app.get_webview_window("status") {
        return Ok(window);
    }
    WebviewWindowBuilder::new(app, "status", WebviewUrl::App("index.html".into()))
        .title(TITLE)
        .inner_size(640., 440.)
        .devtools(false)
        .on_page_load(|window, page| {
            if page.event() == tauri::webview::PageLoadEvent::Finished {
                if let Err(error) = render_status(&window) {
                    eprintln!("{error}");
                }
            }
        })
        .on_new_window(|_, _| NewWindowResponse::Deny)
        .build()
}

fn status(app: &AppHandle, summary: &str, detail: &str) -> tauri::Result<()> {
    app.state::<Mutex<Windows>>().lock().unwrap().status =
        Some((summary.to_owned(), detail.to_owned()));
    let window = status_window(app)?;
    render_status(&window)?;
    focus(&window)
}

fn render_status(window: &WebviewWindow) -> tauri::Result<()> {
    let message = window
        .state::<Mutex<Windows>>()
        .lock()
        .unwrap()
        .status
        .clone();
    if let Some((summary, detail)) = message {
        let summary = json!(summary);
        let detail = json!(detail);
        window.eval(format!(
            "if (document.getElementById('summary')) {{\
               document.getElementById('summary').textContent = {summary};\
               document.getElementById('detail').textContent = {detail};\
             }}"
        ))?;
    }
    Ok(())
}

fn focus(window: &WebviewWindow) -> tauri::Result<()> {
    window.show()?;
    window.unminimize()?;
    window.set_focus()
}

fn refresh_project_session(
    window: &WebviewWindow,
    ticket: &StudioTicket,
    project: &str,
    reconnect: bool,
) -> tauri::Result<()> {
    let url = &ticket.url;
    let current = window.url()?;
    if current.origin() != url.origin() {
        return Err(
            io::Error::other("The service address changed. Close and reopen the project.").into(),
        );
    }
    if current.path() == CONNECT_PATH {
        return window.navigate(url.clone());
    }
    if !studio_menu::is_project_route(&current, project) {
        return Err(io::Error::other("The window is outside its assigned project.").into());
    }
    let token = url
        .fragment()
        .ok_or_else(|| io::Error::other("Missing Studio connection ticket."))?;
    window.eval(project_reconnect_script(ticket, project, token, reconnect))
}

fn project_reconnect_script(
    ticket: &StudioTicket,
    project: &str,
    token: &str,
    reconnect: bool,
) -> String {
    let origin = json!(ticket.url.origin().ascii_serialization());
    let project = json!(project);
    let token = json!(token);
    let generation = json!(ticket.generation);
    format!(
        r#"
        if (location.origin === {origin} && location.pathname === '/'
            && new URLSearchParams(location.search).get('project') === {project}
            && ({reconnect} || window.__FEEDBAX_STUDIO_GENERATION__ !== {generation})) {{
            fetch('/_desktop/exchange', {{method: 'POST', headers: {{'X-Feedbax-Ticket': {token}}}}})
                .then(response => {{
                    if (response.status !== 204 || response.headers.get('X-Feedbax-Project') !== {project})
                        throw new Error('Studio project reauthentication failed');
                    window.__FEEDBAX_STUDIO_GENERATION__ = {generation};
                    dispatchEvent(new Event('studio-native-reconnected'));
                }}).catch(error => console.error(error));
        }}
    "#
    )
}

fn show_studio(
    app: &AppHandle,
    ticket: StudioTicket,
    storage_id: [u8; 16],
    reconnect: bool,
) -> tauri::Result<()> {
    let label = ticket.label();
    let windows = app.state::<Mutex<Windows>>();
    let title = ticket.project.title.as_str();
    if let Some(window) = app.get_webview_window(&label) {
        refresh_project_session(&window, &ticket, &ticket.project.id, reconnect)?;
        window.set_title(title)?;
        focus(&window)?;
    } else {
        let origin = ticket.url.origin();
        #[cfg(target_os = "macos")]
        let frame = native_frame::StudioFrame::new(&ticket.url)?;
        let project_id = ticket.project.id;
        let menu_script = studio_menu::attach(app, &label, &ticket.url, &project_id)?;
        let origin_text = json!(origin.ascii_serialization());
        let generation = json!(ticket.generation);
        let script = format!("{menu_script}; if (location.origin === {origin_text}) window.__FEEDBAX_STUDIO_GENERATION__ = {generation};");
        let builder = WebviewWindowBuilder::new(app, &label, WebviewUrl::External(ticket.url))
            .title(title)
            .inner_size(1350., 910.)
            .min_inner_size(820., 560.)
            .devtools(false)
            .data_store_identifier(storage_id)
            .initialization_script(script)
            .on_navigation(move |url| {
                url.origin() == origin
                    && (url.path() == CONNECT_PATH || studio_menu::is_project_route(url, &project_id))
            })
            .on_new_window(|_, _| NewWindowResponse::Deny);
        #[cfg(target_os = "macos")]
        let builder = {
            let load_frame = frame.clone();
            builder
                .title_bar_style(tauri::TitleBarStyle::Overlay)
                .hidden_title(true)
                .on_page_load(move |window, page| {
                    if page.event() == tauri::webview::PageLoadEvent::Started {
                        studio_menu::clear(window.app_handle(), window.label(), false);
                    }
                    if load_frame.matches(page.url()) {
                        load_frame.refresh(&window);
                    }
                })
        };
        let window = builder
            .build()
            .inspect_err(|_| studio_menu::clear(app, &label, true))?;
        #[cfg(target_os = "macos")]
        if let Err(error) = frame.install(&window) {
            window.destroy()?;
            return Err(error);
        }
    }
    if let Some(window) = app.get_webview_window("status") {
        windows.lock().unwrap().status = None;
        window.destroy()?;
    }
    Ok(())
}

type ServiceResult = Result<(Value, service::Paths), Box<dyn std::error::Error + Send + Sync>>;

fn reserve_operation(app: &AppHandle) -> bool {
    if BUSY.swap(true, Ordering::AcqRel) {
        return false;
    }
    if let Err(error) = studio_menu::set_open_enabled(app, false) {
        eprintln!("{error}");
    }
    true
}

fn release_operation(app: &AppHandle) {
    BUSY.store(false, Ordering::Release);
    if let Err(error) = studio_menu::set_open_enabled(app, true) {
        eprintln!("{error}");
    }
}

fn run_operation(
    app: &AppHandle,
    operation: impl FnOnce() -> ServiceResult + Send + 'static,
    show: impl FnOnce(&AppHandle, Value, service::Paths) -> tauri::Result<()> + Send + 'static,
) {
    if reserve_operation(app) {
        run_reserved_operation(app, operation, show);
    }
}

fn run_reserved_operation(
    app: &AppHandle,
    operation: impl FnOnce() -> ServiceResult + Send + 'static,
    show: impl FnOnce(&AppHandle, Value, service::Paths) -> tauri::Result<()> + Send + 'static,
) {
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let result = operation();
        let callback_app = app.clone();
        let scheduled = app.run_on_main_thread(move || {
            let result = match result {
                Ok((answer, paths)) => {
                    callback_app
                        .state::<Mutex<Windows>>()
                        .lock()
                        .unwrap()
                        .projects_root = Some(paths.projects.clone());
                    show(&callback_app, answer, paths).map_err(|error| error.to_string())
                }
                Err(error) => Err(error.to_string()),
            };
            release_operation(&callback_app);
            if let Err(error) = result {
                if let Err(error) = status(&callback_app, "Studio service action failed.", &error) {
                    eprintln!("{error}");
                }
            }
        });
        if let Err(error) = scheduled {
            BUSY.store(false, Ordering::Release);
            eprintln!("{error}");
        }
    });
}

fn connect_reserved(app: &AppHandle, entry: Entry) {
    let reconnect = matches!(entry, Entry::Reconnect(_));
    run_reserved_operation(
        app,
        move || {
            let release = service::Release::load()?;
            let args: &[&str] = match &entry {
                Entry::NewProject => &[],
                Entry::Folder(folder) => &[
                    "--project-folder",
                    folder
                        .to_str()
                        .ok_or("The selected folder path is not valid Unicode.")?,
                ],
                Entry::Reconnect(id) => &["--project-id", id],
            };
            Ok((release.connect(args)?, release.paths()?))
        },
        move |app, answer, paths| {
            if answer["url"].is_string() {
                let ticket = serde_json::from_value(answer).map_err(io::Error::other)?;
                return show_studio(app, ticket, paths.storage_id, reconnect);
            }
            let summary = match answer["status"].as_str() {
            Some("requires_approval") => {
                "Approve Feedbax Studio in System Settings under Login Items, then select Reconnect."
            },
            Some("release_mismatch") => {
                "A different installed release owns the background service. Use coordinated package replacement."
            },
            Some("installing") => "A package replacement is in progress. Select Reconnect once it finishes.",
            Some("recovery_required") => "Complete the interrupted Studio installation recovery, then select Reconnect.",
            Some("stopping") => "The service is stopping. Select Reconnect once it has stopped.",
            _ => "The Studio service did not become ready. Inspect Service Logs and select Reconnect.",
        };
            status(app, summary, &paths.logs.display().to_string())
        },
    );
}

fn connect(app: &AppHandle, entry: Entry) {
    if reserve_operation(app) {
        connect_reserved(app, entry);
    }
}

fn reconnect(app: &AppHandle) {
    let entry = match studio_menu::focused_webview(app)
        .as_ref()
        .map(|window| window.label())
    {
        Some(label) if label.starts_with(PROJECT_PREFIX) => {
            Entry::Reconnect(label[PROJECT_PREFIX.len()..].to_owned())
        }
        _ => app
            .webview_windows()
            .into_keys()
            .filter_map(|label| label.strip_prefix(PROJECT_PREFIX).map(str::to_owned))
            .min()
            .map(Entry::Reconnect)
            .unwrap_or_default(),
    };
    connect(app, entry);
}

#[cfg(target_os = "macos")]
fn open_project(app: &AppHandle) {
    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSModalResponse, NSModalResponseCancel, NSModalResponseOK, NSOpenPanel};
    use objc2_foundation::{NSString, NSURL};

    if !reserve_operation(app) {
        return;
    }
    let Some(main) = MainThreadMarker::new() else {
        open_failed(app, "Open requires the main thread.");
        return;
    };
    let root = app
        .state::<Mutex<Windows>>()
        .lock()
        .unwrap()
        .projects_root
        .clone();
    let panel = NSOpenPanel::openPanel(main);
    panel.setTitle(Some(&NSString::from_str("Open Studio Project")));
    panel.setCanChooseFiles(false);
    panel.setCanChooseDirectories(true);
    panel.setAllowsMultipleSelection(false);
    panel.setCanCreateDirectories(false);
    panel.setResolvesAliases(false);
    if let Some(root) = root.as_deref().and_then(Path::to_str) {
        panel.setDirectoryURL(Some(&NSURL::fileURLWithPath_isDirectory(
            &NSString::from_str(root),
            true,
        )));
    }

    let app = app.clone();
    let picker = panel.clone();
    let completion = RcBlock::new(move |response: NSModalResponse| {
        picker.orderOut(None);
        if response == NSModalResponseCancel {
            release_operation(&app);
        } else if response != NSModalResponseOK {
            open_failed(&app, "The project folder picker could not open.");
        } else if let Some(path) = picker.URL().and_then(|url| url.path()) {
            connect_reserved(&app, Entry::Folder(PathBuf::from(path.to_string())));
        } else {
            open_failed(&app, "The selected project folder has no local path.");
        }
    });
    panel.beginWithCompletionHandler(&completion);
}

#[cfg(target_os = "macos")]
fn open_failed(app: &AppHandle, detail: &str) {
    release_operation(app);
    if let Err(error) = status(app, "Could not open a Studio project.", detail) {
        eprintln!("{error}");
    }
}

fn stop(app: &AppHandle, disposition: &'static str) {
    run_operation(
        app,
        move || {
            let release = service::Release::load()?;
            Ok((release.stop(disposition)?, release.paths()?))
        },
        |app, answer, _paths| {
            let summary = match answer["status"].as_str() {
                Some("ready_to_stop") => {
                    "The Studio service stopped. Select Reconnect to start it again."
                }
                Some("deferred") => {
                    "The Studio service is still running. Active jobs need a disposition."
                }
                Some("timed_out") => {
                    "Job shutdown reached its deadline. Inspect the jobs or request Force Stop."
                }
                Some("recovery_required") => {
                    "Complete the interrupted Studio installation recovery before stopping the service."
                }
                _ => "Studio could not stop this installed service.",
            };
            status(
                app,
                summary,
                &serde_json::to_string_pretty(&answer).expect("JSON value"),
            )?;
            if answer["status"] == "ready_to_stop" {
                if let Some(window) = app.get_webview_window("studio") {
                    window.destroy()?;
                }
            }
            Ok(())
        },
    );
}

fn setup(app: &mut App) -> tauri::Result<()> {
    app.manage(Mutex::new(Windows::default()));
    let reconnect = MenuItem::with_id(app, "reconnect", "Reconnect", true, None::<&str>)?;
    let stop = MenuItem::with_id(app, "stop", "Stop Service When Idle", true, None::<&str>)?;
    let checkpoint = MenuItem::with_id(
        app,
        "checkpoint",
        "Checkpoint and Stop Service",
        true,
        None::<&str>,
    )?;
    let force = MenuItem::with_id(
        app,
        "force",
        "Force Stop Service (May Lose Unsaved Progress)",
        true,
        None::<&str>,
    )?;
    let permissions =
        MenuItem::with_id(app, "permissions", "Open Login Items…", true, None::<&str>)?;
    let logs = MenuItem::with_id(app, "logs", "Open Service Logs", true, None::<&str>)?;
    let quit = PredefinedMenuItem::quit(app, None)?;
    let studio = Submenu::with_items(
        app,
        TITLE,
        true,
        &[
            &reconnect,
            &stop,
            &checkpoint,
            &force,
            &permissions,
            &logs,
            &quit,
        ],
    )?;
    studio_menu::install(app.handle(), studio)?;
    status_window(app.handle())?;
    connect(app.handle(), Entry::NewProject);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args
        .first()
        .is_some_and(|arg| matches!(arg.as_str(), "install" | "recover" | "uninstall"))
    {
        if let Err(error) =
            service::Release::load().and_then(|release| release.run_installer(&args))
        {
            service::fail(error);
        }
        return;
    }
    if args.first().is_some_and(|arg| arg == "service") {
        if let Err(error) = service::cli(&args[1..]) {
            service::fail(error);
        }
        return;
    }
    let application = tauri::Builder::default()
        .setup(|app| Ok(setup(app)?))
        .invoke_handler(tauri::generate_handler![studio_menu::publish_studio_menu])
        .on_window_event(|window, event| match event {
            tauri::WindowEvent::Focused(_) => {
                studio_menu::refresh_on_main_thread(window.app_handle())
            }
            tauri::WindowEvent::Destroyed => {
                studio_menu::clear(window.app_handle(), window.label(), true);
            }
            _ => {}
        })
        .on_menu_event(|app, event| match event.id().as_ref() {
            "reconnect" => reconnect(app),
            #[cfg(target_os = "macos")]
            studio_menu::OPEN_PROJECT_ID => open_project(app),
            "stop" => stop(app, "defer"),
            "checkpoint" => stop(app, "checkpoint"),
            "force" => stop(app, "force"),
            "permissions" => service::open_login_items(),
            "logs" => run_operation(
                app,
                || {
                    let paths = service::Release::load()?.paths()?;
                    let status = Command::new("/usr/bin/open").arg(&paths.logs).status()?;
                    if !status.success() {
                        return Err(format!(
                            "Could not open the log directory {}",
                            paths.logs.display()
                        )
                        .into());
                    }
                    Ok((Value::Null, paths))
                },
                |_, _, _| Ok(()),
            ),
            id => {
                if let Err(error) = studio_menu::dispatch(app, id) {
                    eprintln!("Studio menu command could not be delivered: {error}");
                }
            }
        })
        .build(tauri::generate_context!());
    match application {
        Ok(app) => app.run(|app, event| {
            if let RunEvent::Reopen {
                has_visible_windows: false,
                ..
            } = event
            {
                if let Err(error) = status_window(app) {
                    eprintln!("{error}");
                }
                reconnect(app);
            }
        }),
        Err(error) => service::fail(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(id: &str, title: &str) -> StudioTicket {
        serde_json::from_value(json!({
            "url": "http://127.0.0.1:41000/_desktop/connect#test-ticket",
            "generation": "a",
            "project": { "id": id, "title": title },
        }))
        .unwrap()
    }

    #[test]
    fn reopening_uses_the_stable_folder_identity() {
        let first = project("first", "Shared title");
        let second = project("second", "Shared title");
        let renamed = project("first", "Renamed title");
        assert_eq!(first.label(), renamed.label());
        assert_ne!(first.label(), second.label());
    }
}

#[cfg(target_os = "macos")]
mod native_frame;
#[cfg(target_os = "macos")]
mod project_document;
mod service;
mod studio_menu;
mod window_session;

use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, VecDeque},
    io,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};
use tauri::{
    menu::{MenuItem, Submenu},
    webview::NewWindowResponse,
    App, AppHandle, Manager, RunEvent, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder,
};
use window_session::{Folder, OpenWindows, RecentProject, RecentProjects, SavedWindow, Size};

static BUSY: AtomicBool = AtomicBool::new(false);
const TITLE: &str = "Feedbax Studio";
const PROJECT_PREFIX: &str = "project-";
const CONNECT_PATH: &str = "/_desktop/connect";

#[derive(Default)]
struct Windows {
    projects_root: Option<PathBuf>,
    status: Option<(String, String)>,
    session_file: Option<PathBuf>,
    recents_file: Option<PathBuf>,
    recents: RecentProjects,
    open: OpenWindows,
    pending: VecDeque<Entry>,
    restore_problems: Vec<String>,
    starting: bool,
    ready: bool,
    waiting: bool,
    flushing_exit: bool,
    closing: BTreeSet<String>,
    quitting: bool,
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
    folder: Folder,
    declaration: PathBuf,
}

impl StudioTicket {
    fn label(&self) -> String {
        format!("{PROJECT_PREFIX}{}", self.project.id)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
enum Entry {
    #[default]
    NewProject,
    Folder(PathBuf),
    Document(PathBuf),
    Reconnect(String, SavedWindow),
    Restore(String, SavedWindow),
    Recent(RecentProject),
}

impl Windows {
    fn queue(&mut self, entry: Entry) {
        if !self.pending.contains(&entry) {
            self.pending.push_back(entry);
        }
    }

    fn restore(&mut self, path: PathBuf, open: OpenWindows) {
        for (id, saved) in &open {
            self.pending
                .push_back(Entry::Restore(id.clone(), saved.clone()));
        }
        self.open = open;
        self.session_file = Some(path);
        self.starting = true;
    }

    fn next(&mut self) -> Option<Entry> {
        if !self.ready || self.waiting || self.quitting || self.flushing_exit {
            return None;
        }
        let startup = std::mem::take(&mut self.starting);
        self.pending
            .pop_front()
            .or_else(|| startup.then_some(Entry::NewProject))
    }

    fn saved_size(&self, id: &str, folder: &Folder) -> Option<Size> {
        self.open
            .get(id)
            .filter(|saved| &saved.folder == folder)
            .map(|saved| saved.size)
    }

    fn opened(&mut self, id: &str, folder: &Folder) {
        self.pending.retain(|entry| {
            !matches!(entry, Entry::Restore(saved_id, saved) if saved_id == id && &saved.folder == folder)
        });
    }

    fn save(&self) -> io::Result<()> {
        if let Some(path) = &self.session_file {
            window_session::write(path, &self.open)?;
        }
        Ok(())
    }

    fn retry(&mut self, entry: Entry) {
        self.pending.push_front(entry);
        self.waiting = true;
    }
}

fn opened_entry(url: &Url) -> Option<Entry> {
    if url.scheme() != "file" || url.has_host() || url.query().is_some() || url.fragment().is_some()
    {
        return None;
    }
    url.to_file_path().ok().map(|path| {
        if !url.path().ends_with('/')
            && path
                .extension()
                .is_some_and(|extension| extension == "feedbax")
        {
            Entry::Document(path)
        } else {
            Entry::Folder(path)
        }
    })
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
    let project_folder = ticket.folder.clone();
    let remembered = windows
        .lock()
        .unwrap()
        .saved_size(&ticket.project.id, &project_folder);
    if let Some(window) = app.get_webview_window(&label) {
        refresh_project_session(&window, &ticket, &ticket.project.id, reconnect)?;
        project_document::ProjectDocuments::bind(
            &window,
            &ticket.project.id,
            ticket.folder.clone(),
            ticket.generation.clone(),
            ticket.declaration.clone(),
        )?;
        window.set_title(title)?;
        focus(&window)?;
    } else {
        let origin = ticket.url.origin();
        #[cfg(target_os = "macos")]
        let frame = native_frame::StudioFrame::new(&ticket.url)?;
        let project_id = ticket.project.id.clone();
        let menu_script = studio_menu::attach(app, &label, &ticket.url, &project_id)?;
        let origin_text = json!(origin.ascii_serialization());
        let generation = json!(ticket.generation);
        let script = format!("{menu_script}; if (location.origin === {origin_text}) window.__FEEDBAX_STUDIO_GENERATION__ = {generation};");
        let size = remembered.unwrap_or(Size::DEFAULT);
        let available = app.primary_monitor()?.map(|monitor| {
            let work = monitor
                .work_area()
                .size
                .to_logical::<f64>(monitor.scale_factor());
            Size {
                width: work.width,
                height: work.height,
            }
        });
        let size = available.map_or(size, |screen| size.bounded(screen));
        let builder = WebviewWindowBuilder::new(app, &label, WebviewUrl::External(ticket.url))
            .title(title)
            .inner_size(size.width, size.height)
            .min_inner_size(Size::MIN.width, Size::MIN.height)
            .devtools(false)
            .data_store_identifier(storage_id)
            .initialization_script(script)
            .on_navigation(move |url| {
                url.origin() == origin
                    && (url.path() == CONNECT_PATH
                        || studio_menu::is_project_route(url, &project_id))
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
        if let Err(error) = project_document::ProjectDocuments::bind(
            &window,
            &ticket.project.id,
            ticket.folder,
            ticket.generation,
            ticket.declaration,
        ) {
            window.destroy()?;
            return Err(error);
        }
        let mut state = windows.lock().unwrap();
        state.open.insert(
            ticket.project.id.clone(),
            SavedWindow {
                folder: project_folder.clone(),
                size,
            },
        );
        state.save()?;
    }
    windows
        .lock()
        .unwrap()
        .opened(&ticket.project.id, &project_folder);
    Ok(())
}

fn record_recent(app: &AppHandle, project: RecentProject) -> tauri::Result<()> {
    let windows = app.state::<Mutex<Windows>>();
    let mut state = windows.lock().unwrap();
    let mut recents = state.recents.clone();
    recents.record(project);
    set_recents(app, &mut state, recents)
}

pub fn rename_recent(app: &AppHandle, id: &str, title: &str) -> tauri::Result<()> {
    let windows = app.state::<Mutex<Windows>>();
    let mut state = windows.lock().unwrap();
    let mut recents = state.recents.clone();
    if recents.rename(id, title) {
        if let Err(error) = set_recents(app, &mut state, recents) {
            drop(state);
            status(
                app,
                "Could not update the recent project name.",
                &error.to_string(),
            )?;
            return Err(error);
        }
    }
    Ok(())
}

fn set_recents(app: &AppHandle, state: &mut Windows, recents: RecentProjects) -> tauri::Result<()> {
    if let Some(path) = &state.recents_file {
        window_session::write(path, &recents)?;
    }
    state.recents = recents;
    studio_menu::set_recent(app, &state.recents)
}

fn clear_recents(app: &AppHandle) -> tauri::Result<()> {
    let windows = app.state::<Mutex<Windows>>();
    let mut state = windows.lock().unwrap();
    set_recents(app, &mut state, RecentProjects::default())
}

type ServiceResult = service::Result<(Value, service::Paths)>;

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
        run_reserved_operation(app, operation, show, "Studio service action failed.", None);
    }
}

fn run_reserved_operation(
    app: &AppHandle,
    operation: impl FnOnce() -> ServiceResult + Send + 'static,
    show: impl FnOnce(&AppHandle, Value, service::Paths) -> tauri::Result<()> + Send + 'static,
    summary: &'static str,
    retry: Option<Entry>,
) {
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let result = operation();
        let callback_app = app.clone();
        let scheduled = app.run_on_main_thread(move || {
            let windows = callback_app.state::<Mutex<Windows>>();
            let mut state = windows.lock().unwrap();
            if state.quitting || state.flushing_exit {
                if !state.quitting {
                    if let Some(entry) = retry {
                        state.pending.push_front(entry);
                    }
                }
                drop(state);
                release_operation(&callback_app);
                return;
            }
            drop(state);
            let result = match result {
                Ok((answer, paths)) => {
                    callback_app
                        .state::<Mutex<Windows>>()
                        .lock()
                        .unwrap()
                        .projects_root = Some(paths.projects.clone());
                    show(&callback_app, answer, paths).map_err(|error| error.to_string())
                }
                Err(error) => {
                    if let Some(entry) = retry {
                        windows.lock().unwrap().retry(entry);
                    }
                    Err(error.to_string())
                }
            };
            release_operation(&callback_app);
            if let Err(error) = result {
                if let Err(error) = status(&callback_app, summary, &error) {
                    eprintln!("{error}");
                }
            }
            drain_entries(&callback_app);
        });
        if let Err(error) = scheduled {
            BUSY.store(false, Ordering::Release);
            eprintln!("{error}");
        }
    });
}

fn connect_reserved(app: &AppHandle, entry: Entry) {
    let reconnect = matches!(entry, Entry::Reconnect(..));
    let request = entry.clone();
    let retry = entry.clone();
    run_reserved_operation(
        app,
        move || {
            let release = service::Release::load()?;
            let args = match &request {
                Entry::NewProject => Vec::new(),
                Entry::Folder(folder) => vec!["--project-folder".into(), path_argument(folder)?],
                Entry::Document(file) => vec!["--project-file".into(), path_argument(file)?],
                Entry::Restore(_, saved) | Entry::Reconnect(_, saved) => {
                    folder_arguments(&saved.folder)?
                }
                Entry::Recent(recent) => folder_arguments(&recent.folder)?,
            };
            let arguments: Vec<_> = args.iter().map(String::as_str).collect();
            let paths = release.paths()?;
            let answer = release.connect(&arguments).unwrap_or_else(
                |error| json!({"status": "connect_failed", "detail": error.to_string()}),
            );
            Ok((answer, paths))
        },
        move |app, answer, paths| {
            if answer["url"].is_string() {
                let ticket: StudioTicket =
                    serde_json::from_value(answer).map_err(io::Error::other)?;
                if let Some((id, folder)) = match &entry {
                    Entry::Restore(id, saved) | Entry::Reconnect(id, saved) => {
                        Some((id, &saved.folder))
                    }
                    Entry::Recent(recent) => Some((&recent.id, &recent.folder)),
                    _ => None,
                } {
                    if ticket.project.id != *id || ticket.folder != *folder {
                        return Err(io::Error::other(
                            "The saved project no longer identifies the same folder.",
                        )
                        .into());
                    }
                }
                let recent = RecentProject {
                    id: ticket.project.id.clone(),
                    title: ticket.project.title.clone(),
                    folder: ticket.folder.clone(),
                };
                show_studio(app, ticket, paths.storage_id, reconnect)?;
                if matches!(
                    entry,
                    Entry::NewProject | Entry::Folder(_) | Entry::Document(_) | Entry::Recent(_)
                ) {
                    record_recent(app, recent)?;
                }
                if let Some(window) = app.get_webview_window("status") {
                    let state = app.state::<Mutex<Windows>>();
                    let mut state = state.lock().unwrap();
                    if state.restore_problems.is_empty() && state.status.take().is_some() {
                        drop(state);
                        window.destroy()?;
                    }
                }
                return Ok(());
            }
            if answer["status"] == "project_refused" {
                let detail = answer["detail"]
                    .as_str()
                    .unwrap_or("The project could not be opened.");
                if let Entry::Restore(_, saved) = entry {
                    let windows = app.state::<Mutex<Windows>>();
                    let problems = {
                        let mut state = windows.lock().unwrap();
                        state
                            .restore_problems
                            .push(format!("{}\n{detail}", saved.folder.path.display()));
                        state.restore_problems.join("\n\n")
                    };
                    return status(app, "Some Studio windows could not be restored.", &problems);
                }
                return status(app, "Could not open a Studio project.", detail);
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
            {
                let windows = app.state::<Mutex<Windows>>();
                windows.lock().unwrap().retry(entry);
            }
            let detail = answer["detail"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| paths.logs.display().to_string());
            status(app, summary, &detail)
        },
        "Could not open a Studio project.",
        Some(retry),
    );
}

fn path_argument(folder: &Path) -> service::Result<String> {
    folder
        .to_str()
        .map(str::to_owned)
        .ok_or("The selected folder path is not valid Unicode.".into())
}

fn folder_arguments(folder: &Folder) -> service::Result<Vec<String>> {
    Ok(vec![
        "--project-folder".into(),
        path_argument(&folder.path)?,
        "--folder-identity".into(),
        folder.identity[0].to_string(),
        folder.identity[1].to_string(),
    ])
}

fn queue_open(app: &AppHandle, entry: Entry) {
    let windows = app.state::<Mutex<Windows>>();
    let mut state = windows.lock().unwrap();
    if state.ready && !state.quitting && !state.flushing_exit {
        state.pending.push_back(entry);
    }
    drop(state);
    drain_entries(app);
}

fn drain_entries(app: &AppHandle) {
    if BUSY.load(Ordering::Acquire) {
        return;
    }
    let entry = app.state::<Mutex<Windows>>().lock().unwrap().next();
    if let Some(entry) = entry {
        if reserve_operation(app) {
            connect_reserved(app, entry);
        }
    }
}

fn remember_size(window: &tauri::Window) -> tauri::Result<()> {
    let Some(id) = window.label().strip_prefix(PROJECT_PREFIX) else {
        return Ok(());
    };
    if window.is_fullscreen()? || window.is_maximized()? || window.is_minimized()? {
        return Ok(());
    }
    let size = window
        .inner_size()?
        .to_logical::<f64>(window.scale_factor()?);
    let state = window.state::<Mutex<Windows>>();
    if let Some(saved) = state.lock().unwrap().open.get_mut(id) {
        saved.size = Size {
            width: size.width,
            height: size.height,
        };
    }
    Ok(())
}

fn close_project(window: &tauri::Window) -> io::Result<()> {
    let Some(id) = window.label().strip_prefix(PROJECT_PREFIX) else {
        return Ok(());
    };
    let windows = window.state::<Mutex<Windows>>();
    let mut state = windows.lock().unwrap();
    if !state.quitting && state.open.remove(id).is_some() {
        state.save()?;
    }
    Ok(())
}

fn quit(app: &AppHandle) {
    let windows = app.state::<Mutex<Windows>>();
    let mut state = windows.lock().unwrap();
    if std::mem::replace(&mut state.quitting, true) {
        return;
    }
    if let Err(error) = state.save() {
        eprintln!("Studio could not save its window session: {error}");
    }
}

#[cfg(target_os = "macos")]
async fn flush_window(window: WebviewWindow) -> Result<(), String> {
    let (sender, receiver) = std::sync::mpsc::channel();
    native_frame::flush_preferences(&window, move |result| {
        let _ = sender.send(result);
    })
    .map_err(|error| error.to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|_| "Window preferences did not finish saving.".to_owned())?
    })
    .await
    .map_err(|error| error.to_string())?
}

#[cfg(target_os = "macos")]
fn resume_window_preferences(window: &WebviewWindow) {
    if let Err(error) = window.eval("window.__FEEDBAX_FLUSH_WINDOW_PREFERENCES__?.(false)") {
        eprintln!("{error}");
    }
}

#[cfg(target_os = "macos")]
fn request_quit(app: &AppHandle) {
    let windows = app.state::<Mutex<Windows>>();
    let mut state = windows.lock().unwrap();
    if state.flushing_exit || state.quitting {
        return;
    }
    state.flushing_exit = true;
    drop(state);
    let open: Vec<_> = app
        .webview_windows()
        .into_iter()
        .filter_map(|(label, window)| label.starts_with(PROJECT_PREFIX).then_some(window))
        .collect();
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let mut pending: Vec<_> = open
            .iter()
            .cloned()
            .map(|window| tauri::async_runtime::spawn(flush_window(window)))
            .collect();
        let mut problem = None;
        for task in pending.drain(..) {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => problem = Some(error),
                Err(error) => problem = Some(error.to_string()),
            }
        }
        let callback = app.clone();
        if let Err(error) = app.run_on_main_thread(move || {
            let windows = callback.state::<Mutex<Windows>>();
            let mut state = windows.lock().unwrap();
            state.flushing_exit = false;
            let result =
                problem.map_or_else(|| state.save().map_err(|error| error.to_string()), Err);
            match result {
                Ok(()) => {
                    state.quitting = true;
                    drop(state);
                    callback.exit(0);
                }
                Err(error) => {
                    drop(state);
                    for window in &open {
                        resume_window_preferences(window);
                    }
                    if let Err(error) = status(
                        &callback,
                        "Could not save the Studio window session.",
                        &error,
                    ) {
                        eprintln!("{error}");
                    }
                    drain_entries(&callback);
                }
            }
        }) {
            eprintln!("{error}");
        }
    });
}

#[cfg(target_os = "macos")]
fn request_project_close(window: &tauri::Window, api: &tauri::CloseRequestApi) {
    let label = window.label();
    if !label.starts_with(PROJECT_PREFIX) {
        return;
    }
    let windows = window.state::<Mutex<Windows>>();
    {
        let mut state = windows.lock().unwrap();
        if state.quitting {
            return;
        }
        api.prevent_close();
        if state.flushing_exit || !state.closing.insert(label.to_owned()) {
            return;
        }
    }
    let app = window.app_handle().clone();
    let Some(webview) = app.get_webview_window(label) else {
        windows.lock().unwrap().closing.remove(label);
        return;
    };
    tauri::async_runtime::spawn(async move {
        let result = flush_window(webview.clone()).await;
        let callback = app.clone();
        if let Err(error) = app.run_on_main_thread(move || {
            let windows = callback.state::<Mutex<Windows>>();
            let mut state = windows.lock().unwrap();
            let label = webview.label();
            state.closing.remove(label);
            if state.flushing_exit || state.quitting {
                return;
            }
            let result = result.and_then(|()| {
                let id = label.strip_prefix(PROJECT_PREFIX).expect("project close");
                let saved = state.open.remove(id);
                if let Err(error) = state.save() {
                    if let Some(saved) = saved {
                        state.open.insert(id.to_owned(), saved);
                    }
                    return Err(error.to_string());
                }
                Ok(())
            });
            drop(state);
            if let Err(error) = result {
                resume_window_preferences(&webview);
                if let Err(error) = status(
                    &callback,
                    "Could not save the Studio window preferences.",
                    &error,
                ) {
                    eprintln!("{error}");
                }
            } else if let Err(error) = webview.destroy() {
                eprintln!("{error}");
            }
        }) {
            eprintln!("{error}");
        }
    });
}

fn start_windows(app: &AppHandle) {
    let result = service::Release::load()
        .and_then(|release| release.paths())
        .and_then(|paths| {
            let path = paths.settings.join("studio-windows.json");
            project_document::ProjectDocuments::start(app, paths.control)?;
            let open = window_session::read(&path)?;
            let recents_path = paths.settings.join("studio-recents.json");
            let recents = window_session::read(&recents_path);
            let windows = app.state::<Mutex<Windows>>();
            let mut state = windows.lock().unwrap();
            state.projects_root = Some(paths.projects);
            state.recents_file = Some(recents_path);
            state.restore(path, open);
            state.ready = true;
            match recents {
                Ok(recents) => {
                    studio_menu::set_recent(app, &recents)?;
                    state.recents = recents;
                }
                Err(error) => {
                    drop(state);
                    status(app, "Could not read recent projects.", &error.to_string())?;
                }
            }
            Ok(())
        });
    if let Err(error) = result {
        if let Err(error) = status(
            app,
            "Studio could not read its window session.",
            &error.to_string(),
        ) {
            eprintln!("{error}");
        }
    }
}

fn reconnect(app: &AppHandle) {
    let windows = app.state::<Mutex<Windows>>();
    {
        let mut state = windows.lock().unwrap();
        if !state.ready {
            drop(state);
            start_windows(app);
            drain_entries(app);
            return;
        }
        if !state.pending.is_empty() {
            state.waiting = false;
            drop(state);
            drain_entries(app);
            return;
        }
    }
    let project = match studio_menu::focused_webview(app)
        .as_ref()
        .map(|window| window.label())
    {
        Some(label) if label.starts_with(PROJECT_PREFIX) => {
            Some(label[PROJECT_PREFIX.len()..].to_owned())
        }
        _ => app
            .webview_windows()
            .into_keys()
            .filter_map(|label| label.strip_prefix(PROJECT_PREFIX).map(str::to_owned))
            .min(),
    };
    let mut state = windows.lock().unwrap();
    let entry = project
        .and_then(|id| {
            state
                .open
                .get(&id)
                .cloned()
                .map(|saved| Entry::Reconnect(id, saved))
        })
        .unwrap_or_default();
    state.pending.push_back(entry);
    drop(state);
    drain_entries(app);
}

#[cfg(target_os = "macos")]
fn open_project(app: &AppHandle) {
    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSModalResponse, NSModalResponseCancel, NSModalResponseOK, NSOpenPanel};
    use objc2_foundation::{NSString, NSURL};

    if app.state::<Mutex<Windows>>().lock().unwrap().flushing_exit || !reserve_operation(app) {
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
            drain_entries(&app);
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
    drain_entries(app);
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
    let quit = MenuItem::with_id(app, "quit", format!("Quit {TITLE}"), true, Some("Cmd+Q"))?;
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
        .manage(Mutex::new(Windows::default()))
        .setup(|app| Ok(setup(app)?))
        .invoke_handler(tauri::generate_handler![
            studio_menu::publish_studio_menu,
            studio_menu::studio_window_action
        ])
        .on_window_event(|window, event| match event {
            #[cfg(target_os = "macos")]
            tauri::WindowEvent::CloseRequested { api, .. } => request_project_close(window, api),
            tauri::WindowEvent::Focused(_) => {
                studio_menu::refresh_on_main_thread(window.app_handle())
            }
            tauri::WindowEvent::Destroyed => {
                project_document::ProjectDocuments::detach(window.app_handle(), window.label());
                studio_menu::clear(window.app_handle(), window.label(), true);
                if let Err(error) = close_project(window) {
                    eprintln!("Studio could not record a closed project: {error}");
                }
            }
            tauri::WindowEvent::Resized(_) | tauri::WindowEvent::ScaleFactorChanged { .. } => {
                if let Err(error) = remember_size(window) {
                    eprintln!("Studio could not read window size: {error}");
                }
            }
            _ => {}
        })
        .on_menu_event(|app, event| match event.id().as_ref() {
            "reconnect" => reconnect(app),
            #[cfg(target_os = "macos")]
            studio_menu::OPEN_PROJECT_ID => open_project(app),
            studio_menu::NEW_PROJECT_ID => queue_open(app, Entry::NewProject),
            studio_menu::CLOSE_PROJECT_ID => {
                if let Some(window) = studio_menu::focused_webview(app)
                    .filter(|window| window.label().starts_with(PROJECT_PREFIX))
                {
                    if let Err(error) = window.close() {
                        eprintln!("{error}");
                    }
                }
            }
            studio_menu::CLEAR_RECENT_ID => {
                if let Err(error) = clear_recents(app) {
                    let _ = status(app, "Could not clear recent projects.", &error.to_string());
                }
            }
            "stop" => stop(app, "defer"),
            "checkpoint" => stop(app, "checkpoint"),
            "force" => stop(app, "force"),
            "permissions" => service::open_login_items(),
            "quit" => {
                #[cfg(target_os = "macos")]
                request_quit(app);
            }
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
                if let Some(recent) = studio_menu::recent_project(app, id) {
                    queue_open(app, Entry::Recent(recent));
                } else if let Err(error) = studio_menu::dispatch(app, id) {
                    eprintln!("Studio menu command could not be delivered: {error}");
                }
            }
        })
        .build(tauri::generate_context!());
    match application {
        Ok(app) => app.run(|app, event| match event {
            RunEvent::Ready => start_windows(app),
            RunEvent::MainEventsCleared => drain_entries(app),
            #[cfg(target_os = "macos")]
            RunEvent::Opened { urls } => {
                let windows = app.state::<Mutex<Windows>>();
                let mut state = windows.lock().unwrap();
                for entry in urls.iter().filter_map(opened_entry) {
                    state.queue(entry);
                }
            }
            RunEvent::Reopen {
                has_visible_windows: false,
                ..
            } => {
                let windows = app.state::<Mutex<Windows>>();
                let state = windows.lock().unwrap();
                let started = state.ready && !state.starting && state.pending.is_empty();
                drop(state);
                if started && !BUSY.load(Ordering::Acquire) {
                    reconnect(app);
                }
            }
            RunEvent::ExitRequested { api, .. } => {
                let state = app.state::<Mutex<Windows>>();
                let should_flush = {
                    let state = state.lock().unwrap();
                    !state.quitting
                        && app
                            .webview_windows()
                            .keys()
                            .any(|label| label.starts_with(PROJECT_PREFIX))
                };
                if should_flush {
                    #[cfg(target_os = "macos")]
                    {
                        api.prevent_exit();
                        request_quit(app);
                    }
                } else {
                    quit(app);
                }
            }
            RunEvent::Exit => quit(app),
            _ => {}
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
            "folder": { "path": format!("/projects/{id}"), "identity": [1, 2] },
            "declaration": format!("/projects/{id}/{title}.feedbax"),
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

    #[test]
    fn finder_queue_precedes_cold_new_and_survives_a_service_pause() {
        let mut state = Windows::default();
        let first =
            opened_entry(&Url::parse("file:///projects/First%20Project/reach.feedbax").unwrap())
                .unwrap();
        assert_eq!(
            first,
            Entry::Document("/projects/First Project/reach.feedbax".into())
        );
        let second =
            opened_entry(&Url::parse("file:///projects/Second.feedbax/").unwrap()).unwrap();
        assert_eq!(second, Entry::Folder("/projects/Second.feedbax/".into()));
        for invalid in [
            "https://example.com/Project",
            "file://remote.example/Project",
            "file:///projects/First?path=other",
        ] {
            assert!(opened_entry(&Url::parse(invalid).unwrap()).is_none());
        }
        state.queue(first.clone());
        assert!(state.next().is_none());
        state.restore("/settings/windows.json".into(), OpenWindows::new());
        state.ready = true;
        state.queue(first.clone());
        state.queue(second.clone());
        let first_entry = state.next().unwrap();
        assert_eq!(first_entry, first);
        state.pending.push_front(first_entry.clone());
        state.waiting = true;
        assert!(state.next().is_none());
        state.waiting = false;
        assert_eq!(state.next(), Some(first_entry));
        assert_eq!(state.next(), Some(second));
        assert!(state.next().is_none());
    }

    #[test]
    fn launch_enqueues_both_saved_windows_and_new_only_for_an_empty_session() {
        let mut state = Windows::default();
        let mut open = OpenWindows::new();
        for (id, width) in [("one", 920.), ("two", 1140.)] {
            let ticket = project(id, id);
            open.insert(
                id.to_owned(),
                SavedWindow {
                    folder: ticket.folder,
                    size: Size {
                        width,
                        height: 620.,
                    },
                },
            );
        }
        state.restore("/settings/windows.json".into(), open.clone());
        state.ready = true;
        for (id, saved) in open {
            assert_eq!(state.next(), Some(Entry::Restore(id, saved)));
        }
        assert!(state.next().is_none());
        state.restore("/settings/windows.json".into(), OpenWindows::new());
        assert_eq!(state.next(), Some(Entry::NewProject));
        assert!(state.next().is_none());
    }

    #[test]
    fn finder_open_reuses_saved_size_and_consumes_only_matching_restore() {
        let mut state = Windows::default();
        let first = project("one", "First");
        let second = project("two", "Second");
        let size = Size {
            width: 1060.,
            height: 650.,
        };
        let first_saved = SavedWindow {
            folder: first.folder.clone(),
            size,
        };
        let second_saved = SavedWindow {
            folder: second.folder,
            size: Size::DEFAULT,
        };
        state.queue(Entry::Folder(first.folder.path.clone()));
        state.restore(
            "/settings/windows.json".into(),
            OpenWindows::from([
                ("one".into(), first_saved),
                ("two".into(), second_saved.clone()),
            ]),
        );
        state.ready = true;
        assert_eq!(state.next(), Some(Entry::Folder(first.folder.path.clone())));
        assert_eq!(state.saved_size("one", &first.folder), Some(size));
        state.opened("one", &first.folder);
        assert_eq!(
            state.next(),
            Some(Entry::Restore("two".into(), second_saved))
        );
        assert!(state.next().is_none());
    }
}

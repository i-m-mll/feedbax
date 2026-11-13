use crate::{
    background_activity::Control, focus, service, studio_menu, Windows, BUSY, CONNECT_PATH,
    PROJECT_PREFIX,
};
use objc2::MainThreadMarker;
use objc2_app_kit::NSApplication;
use serde_json::json;
use std::{
    io,
    sync::{atomic::Ordering, mpsc, Mutex},
    time::Duration,
};
use tauri::{
    webview::NewWindowResponse, AppHandle, Manager, Url, WebviewUrl, WebviewWindowBuilder,
};

const LABEL: &str = "activity";

pub fn observe(app: &AppHandle, activity_launch: bool) {
    let gui = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        if let Err(error) = report(&gui, activity_launch) {
            eprintln!("Studio Activity window registration stopped: {error}");
        }
    });
}

fn report(app: &AppHandle, mut cold_open: bool) -> service::Result<()> {
    let release = service::Release::load()?;
    let paths = release.paths()?;
    let control = Control::new(&paths)?;
    let mut last_error = None;
    loop {
        let (send, receive) = mpsc::sync_channel(1);
        let callback = app.clone();
        app.run_on_main_thread(move || {
            let active = MainThreadMarker::new()
                .is_some_and(|main| NSApplication::sharedApplication(main).isActive());
            let project = active
                .then(|| studio_menu::focused_webview(&callback))
                .flatten()
                .and_then(|window| {
                    window
                        .label()
                        .strip_prefix(PROJECT_PREFIX)
                        .map(str::to_owned)
                });
            let state = callback.state::<Mutex<Windows>>();
            let state = state.lock().unwrap();
            let ready = state.ready && !state.flushing_exit && !BUSY.load(Ordering::Acquire);
            let _ = send.send((project, ready, state.quitting));
        })?;
        let (project, ready, quitting) = receive.recv_timeout(Duration::from_secs(2))?;
        if quitting {
            return Ok(());
        }
        if let Ok(intents) = control.window(project.as_deref(), ready) {
            if let Some(documents) = intents.documents {
                // Launch Services completes under the same lock needed to connect a project.
                if let Ok(activation) = release.activity_activation(&paths) {
                    drop(activation);
                    let (send, receive) = mpsc::sync_channel(1);
                    let callback = app.clone();
                    app.run_on_main_thread(move || {
                        let windows = callback.state::<Mutex<Windows>>();
                        let receipt = windows.lock().unwrap().receive_documents(documents);
                        let _ = send.send(receipt);
                        crate::drain_entries(&callback);
                    })?;
                    if let Some((generation, through)) =
                        receive.recv_timeout(Duration::from_secs(2))?
                    {
                        if let Err(error) = control.acknowledge_documents(&generation, through) {
                            eprintln!("Could not acknowledge Studio document delivery: {error}");
                        }
                    }
                }
            }
            if ready && (cold_open || intents.open.is_some()) {
                match deliver(app, &release, &paths, &control, intents.open) {
                    Ok(()) => {
                        cold_open = false;
                        last_error = None;
                    }
                    Err(error) => {
                        let detail = error.to_string();
                        if last_error.as_ref() != Some(&detail) {
                            eprintln!("Could not open Studio Activity: {detail}");
                            last_error = Some(detail);
                        }
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn deliver(
    app: &AppHandle,
    release: &service::Release,
    paths: &service::Paths,
    control: &Control,
    intent: Option<u64>,
) -> service::Result<()> {
    let (send, receive) = mpsc::sync_channel(1);
    let callback = app.clone();
    app.run_on_main_thread(move || {
        let _ = send.send(open_in_project(&callback).map_err(|error| error.to_string()));
    })?;
    let opened = receive
        .recv_timeout(Duration::from_secs(2))?
        .map_err(io::Error::other)?;
    if !opened {
        let response = release.activity_ticket(paths)?;
        let url = response["url"]
            .as_str()
            .ok_or_else(|| io::Error::other(response.to_string()))?;
        let ticket = url.parse()?;
        let callback = app.clone();
        let storage = paths.storage_id;
        let (send, receive) = mpsc::sync_channel(1);
        app.run_on_main_thread(move || {
            let _ = send.send(show(&callback, ticket, storage).map_err(|error| error.to_string()));
        })?;
        receive
            .recv_timeout(Duration::from_secs(3))?
            .map_err(io::Error::other)?;
    }
    if let Some(intent) = intent {
        control.acknowledge(intent)?;
    }
    Ok(())
}

pub fn reconnect(app: &AppHandle) -> bool {
    if app.get_webview_window(LABEL).is_none() {
        return false;
    }
    if studio_menu::focused_webview(app).is_some_and(|window| window.label() != LABEL) {
        return false;
    }
    crate::run_operation(
        app,
        || {
            let release = service::Release::load()?;
            Ok((release.connect(&["--activity"])?, release.paths()?))
        },
        |app, response, paths| {
            let url = response["url"]
                .as_str()
                .ok_or_else(|| io::Error::other(response.to_string()))?;
            show(
                app,
                url.parse().map_err(io::Error::other)?,
                paths.storage_id,
            )
        },
    );
    true
}

fn open_in_project(app: &AppHandle) -> tauri::Result<bool> {
    if let Some(window) = app.get_webview_window(LABEL) {
        focus(&window)?;
        return Ok(true);
    }
    let window = studio_menu::focused_webview(app)
        .filter(|window| window.label().starts_with(PROJECT_PREFIX))
        .or_else(|| {
            app.webview_windows()
                .into_values()
                .filter(|window| window.label().starts_with(PROJECT_PREFIX))
                .min_by(|left, right| left.label().cmp(right.label()))
        });
    let Some(window) = window else {
        return Ok(false);
    };
    let Some(id) = window.label().strip_prefix(PROJECT_PREFIX) else {
        return Ok(false);
    };
    let url = window.url()?;
    if !studio_menu::is_project_route(&url, id) {
        return Ok(false);
    }
    let origin = json!(url.origin().ascii_serialization());
    let project = json!(id);
    window.eval(format!(
        "if (location.origin === {origin} && location.pathname === '/' && new URLSearchParams(location.search).get('project') === {project}) {{
           document.documentElement.dataset.studioOpenActivity = '1';
           dispatchEvent(new Event('studio-native-open-activity'));
         }}"
    ))?;
    focus(&window)?;
    Ok(true)
}

fn show(app: &AppHandle, ticket: Url, storage: [u8; 16]) -> tauri::Result<()> {
    if let Some(window) = app.get_webview_window(LABEL) {
        window.navigate(ticket)?;
        return focus(&window);
    }
    let origin = ticket.origin();
    let window = WebviewWindowBuilder::new(app, LABEL, WebviewUrl::External(ticket))
        .title("Activity")
        .inner_size(600., 660.)
        .data_store_identifier(storage)
        .devtools(false)
        .on_navigation(move |url| {
            url.origin() == origin
                && (url.path() == CONNECT_PATH
                    || (url.path() == "/" && url.query() == Some("activity=1")))
        })
        .on_new_window(|_, _| NewWindowResponse::Deny)
        .build()?;
    focus(&window)
}

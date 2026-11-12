#[cfg(target_os = "macos")]
mod native_frame;
mod service;

use serde_json::Value;
use std::{
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem, Submenu},
    webview::NewWindowResponse,
    App, AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindow, WebviewWindowBuilder,
};

static BUSY: AtomicBool = AtomicBool::new(false);
const TITLE: &str = "Feedbax Studio";

fn status_window(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    if let Some(window) = app.get_webview_window("status") {
        return Ok(window);
    }
    WebviewWindowBuilder::new(app, "status", WebviewUrl::App("index.html".into()))
        .title(TITLE)
        .inner_size(640., 440.)
        .devtools(false)
        .on_new_window(|_, _| NewWindowResponse::Deny)
        .build()
}

fn status(app: &AppHandle, summary: &str, detail: &str) -> tauri::Result<()> {
    let window = status_window(app)?;
    let summary = serde_json::to_string(summary).expect("serialize string");
    let detail = serde_json::to_string(detail).expect("serialize string");
    window.eval(format!(
        "document.getElementById('summary').textContent = {summary};\
         document.getElementById('detail').textContent = {detail};"
    ))?;
    window.set_focus()
}

fn show_studio(app: &AppHandle, url: &str, paths: service::Paths) -> tauri::Result<()> {
    let url = url
        .parse::<tauri::Url>()
        .map_err(tauri::Error::InvalidUrl)?;
    if let Some(window) = app.get_webview_window("studio") {
        window.navigate(url)?;
        window.set_focus()?;
    } else {
        let origin = url.origin();
        #[cfg(target_os = "macos")]
        let frame = native_frame::StudioFrame::new(&url)?;
        let builder = WebviewWindowBuilder::new(app, "studio", WebviewUrl::External(url))
            .title(TITLE)
            .inner_size(1350., 910.)
            .min_inner_size(820., 560.)
            .devtools(false)
            .data_store_identifier(paths.storage_id)
            .on_navigation(move |url| url.origin() == origin)
            .on_new_window(|_, _| NewWindowResponse::Deny);
        #[cfg(target_os = "macos")]
        let builder = {
            let load_frame = frame.clone();
            builder
                .title_bar_style(tauri::TitleBarStyle::Overlay)
                .hidden_title(true)
                .on_page_load(move |window, page| {
                    if load_frame.matches(page.url()) {
                        load_frame.refresh(&window);
                    }
                })
        };
        let window = builder.build()?;
        #[cfg(target_os = "macos")]
        if let Err(error) = frame.install(&window) {
            window.destroy()?;
            return Err(error);
        }
    }
    if let Some(window) = app.get_webview_window("status") {
        window.destroy()?;
    }
    Ok(())
}

fn run_operation(
    app: &AppHandle,
    operation: impl FnOnce() -> std::result::Result<
            (Value, service::Paths),
            Box<dyn std::error::Error + Send + Sync>,
        > + Send
        + 'static,
    show: impl FnOnce(&AppHandle, Value, service::Paths) -> tauri::Result<()> + Send + 'static,
) {
    if BUSY.swap(true, Ordering::AcqRel) {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let result = operation();
        let callback_app = app.clone();
        let scheduled = app.run_on_main_thread(move || {
            let result = match result {
                Ok((answer, paths)) => show(&callback_app, answer, paths),
                Err(error) => status(
                    &callback_app,
                    "Studio service action failed.",
                    &error.to_string(),
                ),
            };
            BUSY.store(false, Ordering::Release);
            if let Err(error) = result {
                eprintln!("{error}");
            }
        });
        if let Err(error) = scheduled {
            BUSY.store(false, Ordering::Release);
            eprintln!("{error}");
        }
    });
}

fn connect(app: &AppHandle) {
    run_operation(
        app,
        || {
            let release = service::Release::load()?;
            Ok((release.connect()?, release.paths()?))
        },
        |app, answer, paths| {
            if let Some(url) = answer["url"].as_str() {
                return show_studio(app, url, paths);
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
    let edit = Submenu::with_items(
        app,
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(app, None)?,
            &PredefinedMenuItem::redo(app, None)?,
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
        ],
    )?;
    app.set_menu(Menu::with_items(app, &[&studio, &edit])?)?;
    status_window(app.handle())?;
    connect(app.handle());
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
        .on_menu_event(|app, event| match event.id().as_ref() {
            "reconnect" => connect(app),
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
            _ => {}
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
                connect(app);
            }
        }),
        Err(error) => service::fail(error),
    }
}

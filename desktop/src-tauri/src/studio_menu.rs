use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Mutex};
use tauri::{
    ipc::CapabilityBuilder,
    menu::{Menu, MenuItem, PredefinedMenuItem, Submenu},
    AppHandle, Manager, Url, WebviewWindow,
};

const PREFIX: &str = "studio-command:";
pub const OPEN_PROJECT_ID: &str = "open-project";

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    scope: String,
    text_editing: bool,
    sections: Vec<Section>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum Group {
    Edit,
    View,
    Run,
    Plan,
}

impl Group {
    fn title(self) -> &'static str {
        match self {
            Self::Edit => "Edit",
            Self::View => "View",
            Self::Run => "Run",
            Self::Plan => "Plan",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct Section {
    title: Group,
    items: Vec<Item>,
}

#[derive(Clone, Debug, Deserialize)]
struct Item {
    id: String,
    label: String,
    accelerator: Option<String>,
    enabled: bool,
}

struct StudioWindow {
    origin: String,
    project_id: String,
    snapshot: Option<Snapshot>,
}

impl StudioWindow {
    fn permits(&self, url: &Url) -> bool {
        url.origin().ascii_serialization() == self.origin && is_project_route(url, &self.project_id)
    }
}

pub fn is_project_route(url: &Url, project: &str) -> bool {
    url.path() == "/"
        && url
            .query_pairs()
            .find(|(key, _)| key == "project")
            .is_some_and(|(_, value)| value == project)
}

struct Presented {
    window: String,
    scope: String,
    menu_number: u64,
}

#[derive(Serialize)]
struct Command<'a> {
    scope: &'a str,
    id: &'a str,
}

#[derive(Default)]
struct Routing {
    windows: HashMap<String, StudioWindow>,
    presented: Option<Presented>,
    menu_number: u64,
}

impl Routing {
    fn destination(&self, focused: &str, menu_id: &str) -> Option<(&str, Command<'_>)> {
        let shown = self.presented.as_ref()?;
        let target = self.windows.get(focused)?;
        let snapshot = target.snapshot.as_ref()?;
        let (number, id) = menu_id.strip_prefix(PREFIX)?.split_once(':')?;
        if shown.window != focused
            || shown.scope != snapshot.scope
            || number.parse::<u64>().ok()? != shown.menu_number
        {
            return None;
        }
        let item = snapshot
            .sections
            .iter()
            .filter(|section| !snapshot.text_editing || !matches!(section.title, Group::Edit))
            .flat_map(|section| &section.items)
            .find(|item| item.id == id && item.enabled)?;
        Some((
            &target.origin,
            Command {
                scope: &snapshot.scope,
                id: &item.id,
            },
        ))
    }
}

struct StudioMenu {
    service: Submenu<tauri::Wry>,
    file: Submenu<tauri::Wry>,
    open: MenuItem<tauri::Wry>,
    routing: Mutex<Routing>,
}

fn edit_menu(
    app: &AppHandle,
    menu_number: u64,
    commands: Option<&Section>,
) -> tauri::Result<Submenu<tauri::Wry>> {
    let edit = Submenu::new(app, Group::Edit.title(), true)?;
    if let Some(commands) = commands {
        append_commands(app, &edit, menu_number, commands)?;
    } else {
        edit.append_items(&[
            &PredefinedMenuItem::undo(app, None)?,
            &PredefinedMenuItem::redo(app, None)?,
        ])?;
    }
    edit.append_items(&[
        &PredefinedMenuItem::cut(app, None)?,
        &PredefinedMenuItem::copy(app, None)?,
        &PredefinedMenuItem::paste(app, None)?,
        &PredefinedMenuItem::select_all(app, None)?,
    ])?;
    Ok(edit)
}

fn append_commands(
    app: &AppHandle,
    menu: &Submenu<tauri::Wry>,
    menu_number: u64,
    section: &Section,
) -> tauri::Result<()> {
    for item in &section.items {
        menu.append(&MenuItem::with_id(
            app,
            format!("{PREFIX}{menu_number}:{}", item.id),
            &item.label,
            item.enabled,
            item.accelerator.as_deref(),
        )?)?;
    }
    Ok(())
}

fn build_menu(
    app: &AppHandle,
    service: &Submenu<tauri::Wry>,
    file: &Submenu<tauri::Wry>,
    snapshot: Option<&Snapshot>,
    menu_number: u64,
) -> tauri::Result<Menu<tauri::Wry>> {
    let edit = snapshot
        .filter(|snapshot| !snapshot.text_editing)
        .and_then(|snapshot| {
            snapshot
                .sections
                .iter()
                .find(|section| matches!(section.title, Group::Edit))
        });
    let menu = Menu::with_items(app, &[service, file, &edit_menu(app, menu_number, edit)?])?;
    if let Some(snapshot) = snapshot {
        for section in snapshot
            .sections
            .iter()
            .filter(|section| !matches!(section.title, Group::Edit))
        {
            let submenu = Submenu::new(app, section.title.title(), true)?;
            append_commands(app, &submenu, menu_number, section)?;
            menu.append(&submenu)?;
        }
    }
    Ok(menu)
}

pub fn install(app: &AppHandle, service: Submenu<tauri::Wry>) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, OPEN_PROJECT_ID, "Open…", true, Some("CmdOrCtrl+O"))?;
    let file = Submenu::with_items(app, "File", true, &[&open])?;
    app.set_menu(build_menu(app, &service, &file, None, 0)?)?;
    app.manage(StudioMenu {
        service,
        file,
        open,
        routing: Mutex::new(Routing::default()),
    });
    Ok(())
}

pub fn set_open_enabled(app: &AppHandle, enabled: bool) -> tauri::Result<()> {
    app.state::<StudioMenu>().open.set_enabled(enabled)
}

pub fn attach(
    app: &AppHandle,
    label: &str,
    url: &Url,
    project_id: &str,
) -> tauri::Result<String> {
    let origin = url.origin().ascii_serialization();
    app.add_capability(capability(label, &origin))?;
    app.state::<StudioMenu>()
        .routing
        .lock()
        .unwrap()
        .windows
        .insert(
            label.to_string(),
            StudioWindow {
                origin: origin.clone(),
                project_id: project_id.to_owned(),
                snapshot: None,
            },
        );
    let origin = serde_json::json!(origin);
    Ok(format!(
        "if (location.origin === {origin}) {{\
           Object.defineProperty(window, '__FEEDBAX_STUDIO_MENU__', {{\
             value: Object.freeze({{publish: snapshot =>\
               window.__TAURI_INTERNALS__.invoke('publish_studio_menu', {{snapshot}})}})\
           }});\
         }}"
    ))
}

fn capability(label: &str, origin: &str) -> CapabilityBuilder {
    CapabilityBuilder::new(format!("studio-menu-{label}"))
        .local(false)
        .remote(format!("{origin}/*"))
        .webview(label)
        .permission("allow-publish-studio-menu")
}

pub fn focused_webview(app: &AppHandle) -> Option<WebviewWindow> {
    app.webview_windows()
        .into_values()
        .find(|window| window.is_focused().unwrap_or(false))
}

fn refresh(app: &AppHandle) -> tauri::Result<()> {
    let focused = focused_webview(app);
    let state = app.state::<StudioMenu>();
    let mut routing = state.routing.lock().unwrap();
    let target = focused.as_ref().and_then(|window| {
        let snapshot = routing.windows.get(window.label())?.snapshot.as_ref()?;
        Some((window.label(), snapshot))
    });
    let same = match (target, routing.presented.as_ref()) {
        (Some((window, snapshot)), Some(shown)) => {
            shown.window == window && shown.scope == snapshot.scope
        }
        (None, None) => true,
        _ => false,
    };
    if same {
        return Ok(());
    }
    let menu_number = routing.menu_number + 1;
    let menu = build_menu(
        app,
        &state.service,
        &state.file,
        target.map(|(_, snapshot)| snapshot),
        menu_number,
    )?;
    let presented = target.map(|(window, snapshot)| Presented {
        window: window.to_string(),
        scope: snapshot.scope.clone(),
        menu_number,
    });
    app.set_menu(menu)?;
    routing.presented = presented;
    routing.menu_number = menu_number;
    Ok(())
}

pub fn refresh_on_main_thread(app: &AppHandle) {
    let target = app.clone();
    if let Err(error) = app.run_on_main_thread(move || {
        if let Err(error) = refresh(&target) {
            eprintln!("Studio menu could not be updated: {error}");
        }
    }) {
        eprintln!("Studio menu could not be scheduled: {error}");
    }
}

pub fn clear(app: &AppHandle, label: &str, destroyed: bool) {
    let state = app.state::<StudioMenu>();
    let mut routing = state.routing.lock().unwrap();
    if destroyed {
        routing.windows.remove(label);
    } else if let Some(window) = routing.windows.get_mut(label) {
        window.snapshot = None;
    }
    drop(routing);
    refresh_on_main_thread(app);
}

#[tauri::command]
pub fn publish_studio_menu(
    window: WebviewWindow,
    snapshot: Option<Snapshot>,
) -> Result<(), String> {
    let url = window.url().map_err(|error| error.to_string())?;
    let state = window.state::<StudioMenu>();
    let mut routing = state.routing.lock().unwrap();
    let target = routing
        .windows
        .get_mut(window.label())
        .filter(|target| target.permits(&url))
        .ok_or("Studio menu requires its registered service origin and project window")?;
    target.snapshot = snapshot;
    drop(routing);
    refresh_on_main_thread(window.app_handle());
    Ok(())
}

pub fn dispatch(app: &AppHandle, menu_id: &str) -> tauri::Result<()> {
    if !menu_id.starts_with(PREFIX) {
        return Ok(());
    }
    let Some(window) = focused_webview(app) else {
        return Ok(());
    };
    let url = window.url()?;
    let script = {
        let state = app.state::<StudioMenu>();
        let routing = state.routing.lock().unwrap();
        let Some((origin, command)) = routing.destination(window.label(), menu_id) else {
            return Ok(());
        };
        let target = &routing.windows[window.label()];
        if !target.permits(&url) {
            return Ok(());
        }
        let origin = serde_json::json!(origin);
        let command = serde_json::json!(command);
        let project = serde_json::json!(target.project_id);
        format!("if (location.origin === {origin} && location.pathname === '/' && new URLSearchParams(location.search).get('project') === {project}) dispatchEvent(new CustomEvent('studio-native-command', {{detail: {command}}}));")
    };
    window.eval(script)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::{
        ipc::RuntimeCapability,
        utils::acl::{capability::CapabilityFile, RemoteUrlPattern},
    };

    #[test]
    fn permission_is_limited_to_the_attached_webview_and_exact_service_origin() {
        let CapabilityFile::Capability(permission) =
            capability("studio-one", "http://127.0.0.1:41000").build()
        else {
            panic!("expected one capability")
        };
        assert!(!permission.local);
        assert!(permission.windows.is_empty());
        assert_eq!(permission.webviews, ["studio-one"]);
        assert_eq!(permission.permissions.len(), 1);
        let [pattern]: [String; 1] = permission.remote.unwrap().urls.try_into().unwrap();
        let pattern: RemoteUrlPattern = pattern.parse().unwrap();
        assert!(pattern.test(&"http://127.0.0.1:41000/?project=one".parse().unwrap()));
        for url in [
            "http://127.0.0.1:41001/",
            "https://127.0.0.1:41000/",
            "http://localhost:41000/",
        ] {
            assert!(!pattern.test(&url.parse().unwrap()));
        }
    }

    fn snapshot(scope: &str) -> Snapshot {
        serde_json::from_value(serde_json::json!({
            "scope": scope,
            "textEditing": false,
            "sections": [
                { "title": "Run", "items": [
                    { "id": "run", "label": "Run", "accelerator": "Cmd+R", "enabled": true },
                    { "id": "stop", "label": "Stop", "accelerator": "Cmd+.", "enabled": false },
                ] },
                { "title": "Edit", "items": [
                    { "id": "undo", "label": "Undo", "accelerator": "Cmd+Z", "enabled": true },
                ] },
            ],
        }))
        .unwrap()
    }

    #[test]
    fn managed_window_publishes_commands_only_at_its_own_project_route() {
        let window = StudioWindow {
            origin: "http://127.0.0.1:41000".into(),
            project_id: "one".into(),
            snapshot: None,
        };
        assert!(window.permits(&"http://127.0.0.1:41000/?project=one".parse().unwrap()));
        for url in [
            "http://127.0.0.1:41000/",
            "http://127.0.0.1:41000/?project=two",
            "http://127.0.0.1:41000/_desktop/connect#ticket",
            "http://127.0.0.1:41001/?project=one",
        ] {
            assert!(!window.permits(&url.parse().unwrap()));
        }
    }

    #[test]
    fn delivers_only_the_enabled_command_from_the_focused_menu_generation() {
        let mut routing = Routing::default();
        for (label, scope) in [
            ("studio-one", "first-project"),
            ("studio-two", "other-project"),
        ] {
            routing.windows.insert(
                label.into(),
                StudioWindow {
                    origin: "http://127.0.0.1:41000".into(),
                    project_id: scope.into(),
                    snapshot: Some(snapshot(scope)),
                },
            );
        }
        routing.presented = Some(Presented {
            window: "studio-one".into(),
            scope: "first-project".into(),
            menu_number: 2,
        });
        let (origin, command) = routing
            .destination("studio-one", "studio-command:2:run")
            .unwrap();
        assert_eq!(
            (origin, command.scope, command.id),
            ("http://127.0.0.1:41000", "first-project", "run")
        );
        assert!(routing
            .destination("studio-one", "studio-command:2:stop")
            .is_none());
        assert!(routing
            .destination("studio-two", "studio-command:2:run")
            .is_none());
        assert!(routing
            .destination("status", "studio-command:2:run")
            .is_none());
        assert!(routing
            .destination("studio-one", "studio-command:1:run")
            .is_none());
        assert!(routing.destination("studio-one", "stop").is_none());

        let target = routing.windows.get_mut("studio-one").unwrap();
        target.snapshot = Some(snapshot("new-project"));
        assert!(routing
            .destination("studio-one", "studio-command:2:run")
            .is_none());
        routing.presented = Some(Presented {
            window: "studio-one".into(),
            scope: "new-project".into(),
            menu_number: 3,
        });
        routing
            .windows
            .get_mut("studio-one")
            .unwrap()
            .snapshot
            .as_mut()
            .unwrap()
            .text_editing = true;
        assert!(routing
            .destination("studio-one", "studio-command:3:undo")
            .is_none());
        routing.windows.get_mut("studio-one").unwrap().snapshot = None;
        assert!(routing
            .destination("studio-one", "studio-command:3:run")
            .is_none());
    }
}

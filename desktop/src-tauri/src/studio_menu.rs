use crate::window_session::{RecentProject, RecentProjects};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Mutex};
use tauri::{
    ipc::CapabilityBuilder,
    menu::{Menu, MenuItem, PredefinedMenuItem, Submenu},
    AppHandle, Manager, Url, WebviewWindow,
};

const PREFIX: &str = "studio-command:";
pub const OPEN_PROJECT_ID: &str = "open-project";
pub const NEW_PROJECT_ID: &str = "new-project";
pub const CLOSE_PROJECT_ID: &str = "close-project";
pub const CLEAR_RECENT_ID: &str = "clear-recent-projects";
const RECENT_PREFIX: &str = "recent-project:";

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    scope: String,
    text_editing: bool,
    sections: Vec<Section>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum Group {
    File,
    Edit,
    View,
    Run,
    Plan,
}

impl Group {
    fn title(self) -> &'static str {
        match self {
            Self::File => "File",
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
            .filter(|(key, _)| key == "project")
            .map(|(_, value)| value)
            .eq([project])
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
    new: MenuItem<tauri::Wry>,
    close: MenuItem<tauri::Wry>,
    recent: Submenu<tauri::Wry>,
    recents: Mutex<RecentMenu>,
    routing: Mutex<Routing>,
}

#[derive(Default)]
struct RecentMenu {
    generation: u64,
    entries: Vec<RecentProject>,
}

impl RecentMenu {
    fn id(&self, index: usize) -> String {
        format!("{RECENT_PREFIX}{}:{index}", self.generation)
    }

    fn resolve(&self, command: &str) -> Option<&RecentProject> {
        let (generation, index) = command.strip_prefix(RECENT_PREFIX)?.split_once(':')?;
        if generation.parse::<u64>().ok()? != self.generation {
            return None;
        }
        self.entries.get(index.parse::<usize>().ok()?)
    }
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
    if !commands.is_some_and(|section| section.items.iter().any(|item| item.id == "modelSelectAll"))
    {
        edit.append_items(&[
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
        ])?;
    }
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
    let file_commands = snapshot.and_then(|snapshot| {
        snapshot
            .sections
            .iter()
            .find(|section| matches!(section.title, Group::File))
    });
    let file_menu = Submenu::new(app, Group::File.title(), true)?;
    for item in file.items()? {
        file_menu.append(&item)?;
    }
    if let Some(commands) = file_commands {
        append_commands(app, &file_menu, menu_number, commands)?;
    }
    let menu = Menu::with_items(
        app,
        &[service, &file_menu, &edit_menu(app, menu_number, edit)?],
    )?;
    if let Some(snapshot) = snapshot {
        for section in snapshot
            .sections
            .iter()
            .filter(|section| !matches!(section.title, Group::Edit | Group::File))
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
    let new = MenuItem::with_id(app, NEW_PROJECT_ID, "New", true, Some("CmdOrCtrl+N"))?;
    let close = MenuItem::with_id(app, CLOSE_PROJECT_ID, "Close", false, Some("CmdOrCtrl+W"))?;
    let recent = Submenu::new(app, "Open Recent", false)?;
    let file = Submenu::with_items(app, "File", true, &[&new, &open, &recent, &close])?;
    app.set_menu(build_menu(app, &service, &file, None, 0)?)?;
    app.manage(StudioMenu {
        service,
        file,
        open,
        new,
        close,
        recent,
        recents: Mutex::new(RecentMenu::default()),
        routing: Mutex::new(Routing::default()),
    });
    Ok(())
}

pub fn set_open_enabled(app: &AppHandle, enabled: bool) -> tauri::Result<()> {
    let state = app.state::<StudioMenu>();
    state.open.set_enabled(enabled)?;
    state.new.set_enabled(enabled)?;
    state
        .recent
        .set_enabled(enabled && !state.recents.lock().unwrap().entries.is_empty())
}

pub fn set_recent(app: &AppHandle, projects: &RecentProjects) -> tauri::Result<()> {
    let state = app.state::<StudioMenu>();
    let mut menu = state.recents.lock().unwrap();
    if menu.entries == projects.entries() {
        return Ok(());
    }
    let next = RecentMenu {
        generation: menu.generation + 1,
        entries: projects.entries().to_vec(),
    };
    let items = next
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            MenuItem::with_id(app, next.id(index), &entry.title, true, None::<&str>)
        })
        .collect::<tauri::Result<Vec<_>>>()?;
    for item in state.recent.items()? {
        state.recent.remove(&item)?;
    }
    for item in &items {
        state.recent.append(item)?;
    }
    state.recent.append(&PredefinedMenuItem::separator(app)?)?;
    state.recent.append(&MenuItem::with_id(
        app,
        CLEAR_RECENT_ID,
        "Clear Menu",
        true,
        None::<&str>,
    )?)?;
    state
        .recent
        .set_enabled(state.open.is_enabled()? && !next.entries.is_empty())?;
    *menu = next;
    Ok(())
}

pub fn recent_project(app: &AppHandle, command: &str) -> Option<RecentProject> {
    app.state::<StudioMenu>()
        .recents
        .lock()
        .unwrap()
        .resolve(command)
        .cloned()
}

pub fn attach(app: &AppHandle, label: &str, url: &Url, project_id: &str) -> tauri::Result<String> {
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
               window.__TAURI_INTERNALS__.invoke('publish_studio_menu', {{snapshot}}),\
               windowAction: action => window.__TAURI_INTERNALS__.invoke('studio_window_action', {{action}})}})\
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
        .permission("allow-studio-window-action")
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
    let closable = focused
        .as_ref()
        .is_some_and(|window| routing.windows.contains_key(window.label()));
    state.close.set_enabled(closable)?;
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

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum WindowAction {
    Drag {},
    Zoom {},
    Title { title: String },
}

#[tauri::command]
pub async fn studio_window_action(
    window: WebviewWindow,
    action: WindowAction,
) -> Result<(), String> {
    let (reply, result) = tauri::async_runtime::channel(1);
    let app = window.app_handle().clone();
    app.run_on_main_thread(move || {
        let _ = reply.try_send(apply_window_action(&window, action));
    })
    .map_err(|error| error.to_string())?;
    let mut result = result;
    result
        .recv()
        .await
        .ok_or_else(|| "Studio window action was not completed".to_owned())?
}

fn apply_window_action(window: &WebviewWindow, action: WindowAction) -> Result<(), String> {
    let url = window.url().map_err(|error| error.to_string())?;
    {
        let state = window.state::<StudioMenu>();
        let routing = state.routing.lock().unwrap();
        routing
            .windows
            .get(window.label())
            .filter(|target| target.permits(&url))
            .ok_or(
                "Studio window action requires its registered service origin and project window",
            )?;
    }
    let run = || -> tauri::Result<()> {
        match action {
            WindowAction::Title { title } if valid_project_title(&title) => {
                crate::project_document::ProjectDocuments::refresh(window);
            }
            WindowAction::Title { .. } => {
                return Err(std::io::Error::other("Invalid Studio project title").into());
            }
            action if window.is_focused()? && !window.is_fullscreen()? => match action {
                WindowAction::Drag {} => window.start_dragging()?,
                WindowAction::Zoom {} if window.is_resizable()? => {
                    if window.is_maximized()? {
                        window.unmaximize()?
                    } else {
                        window.maximize()?
                    }
                }
                _ => {}
            },
            _ => {}
        }
        Ok(())
    };
    run().map_err(|error| error.to_string())
}

fn valid_project_title(title: &str) -> bool {
    let mut bytes = title.bytes();
    bytes
        .next()
        .is_some_and(|first| first.is_ascii_alphanumeric() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
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
        let ids: Vec<_> = permission
            .permissions
            .iter()
            .map(|p| p.identifier().get())
            .collect();
        assert_eq!(
            ids,
            ["allow-publish-studio-menu", "allow-studio-window-action"]
        );
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
            "http://127.0.0.1:41000/?project=one&project=two",
            "http://127.0.0.1:41000/_desktop/connect#ticket",
            "http://127.0.0.1:41001/?project=one",
        ] {
            assert!(!window.permits(&url.parse().unwrap()));
        }
    }

    #[test]
    fn window_action_cannot_name_another_target() {
        for kind in ["drag", "zoom"] {
            assert!(
                serde_json::from_value::<WindowAction>(serde_json::json!({"kind": kind})).is_ok()
            );
        }
        for action in [
            serde_json::json!({"kind":"drag","label":"another-window"}),
            serde_json::json!({"kind":"title","title":"project","path":"/another/project"}),
            serde_json::json!({"kind":"close"}),
        ] {
            assert!(
                serde_json::from_value::<WindowAction>(action.clone()).is_err(),
                "{action}"
            );
        }
        assert!(valid_project_title("untitled-0123"));
        assert!(!valid_project_title("invalid\nTitle"));
        assert!(!valid_project_title("-title"));
    }

    #[test]
    fn recent_menu_events_cannot_select_a_reordered_entry() {
        let entry = RecentProject {
            id: "project".into(),
            title: "Project".into(),
            folder: crate::window_session::Folder {
                path: "/project".into(),
                identity: [1, 2],
            },
        };
        let menu = RecentMenu {
            generation: 2,
            entries: vec![entry.clone()],
        };
        assert_eq!(menu.resolve(&menu.id(0)), Some(&entry));
        assert!(menu.resolve("recent-project:1:0").is_none());
        assert!(menu.resolve("recent-project:2:1").is_none());
        assert!(RecentMenu {
            generation: 3,
            entries: vec![]
        }
        .resolve(&menu.id(0))
        .is_none());
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

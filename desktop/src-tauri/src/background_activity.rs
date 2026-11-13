use crate::{open_path, service, window_session};
use block2::RcBlock;
use objc2::{
    define_class,
    rc::Retained,
    runtime::{NSObject, NSObjectProtocol, ProtocolObject},
    sel, DefinedClass, MainThreadMarker, MainThreadOnly,
};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSEvent,
    NSEventModifierFlags, NSEventType, NSMenu, NSMenuItem, NSRunningApplication, NSStatusBar,
    NSStatusItem, NSVariableStatusItemLength, NSWorkspace, NSWorkspaceOpenConfiguration,
};
use objc2_foundation::{NSArray, NSPoint, NSRunLoop, NSString, NSTimer, NSURL};
use objc2_user_notifications::{
    UNAuthorizationOptions, UNAuthorizationStatus, UNMutableNotificationContent, UNNotification,
    UNNotificationRequest, UNUserNotificationCenter,
};
use reqwest::blocking::{Client, RequestBuilder};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    fs::File,
    path::{Path, PathBuf},
    process::{Child, ExitStatus},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

const PREFIX: &str = "http://localhost/_desktop/activity";
pub const OPEN_DOCUMENTS: &str = "--open-documents";
const OPEN_ACTIVITY: &str = "--activity";

#[derive(Deserialize)]
struct GuiTarget {
    pid: Option<i32>,
}

#[derive(Deserialize)]
pub struct WindowIntents {
    pub open: Option<u64>,
    pub documents: Option<DocumentIntents>,
}

#[derive(Deserialize)]
pub struct DocumentIntents {
    pub generation: String,
    pub batches: Vec<DocumentOpen>,
}

#[derive(Deserialize, Serialize)]
pub struct DocumentOpen {
    pub sequence: u64,
    pub urls: Vec<tauri::Url>,
}

pub struct Control(Client);

impl Control {
    pub fn new(paths: &service::Paths) -> service::Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-feedbax-release", paths.release.parse()?);
        Ok(Self(
            Client::builder()
                .unix_socket(paths.control.join("service.sock"))
                .default_headers(headers)
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(3))
                .build()?,
        ))
    }

    fn send<T: DeserializeOwned>(request: RequestBuilder) -> service::Result<T> {
        Ok(request.send()?.error_for_status()?.json()?)
    }

    fn snapshot(&self, after: Option<u64>) -> service::Result<Snapshot> {
        let url = after.map_or_else(
            || PREFIX.to_owned(),
            |cursor| format!("{PREFIX}?after={cursor}"),
        );
        Self::send(self.0.get(url))
    }

    fn open(&self, activity: bool) -> service::Result<Option<i32>> {
        let request = if activity {
            self.0.post(format!("{PREFIX}/open"))
        } else {
            self.0.get(format!("{PREFIX}/window"))
        };
        Ok(Self::send::<GuiTarget>(request)?.pid)
    }

    fn open_documents(&self, documents: &DocumentOpen) -> service::Result<Option<i32>> {
        let request = self
            .0
            .post(format!("{PREFIX}/open-documents"))
            .json(documents);
        Ok(Self::send::<GuiTarget>(request)?.pid)
    }

    fn stop(&self, force: bool) -> service::Result<()> {
        let mode = if force { "force" } else { "checkpoint" };
        let result: Vec<Value> = Self::send(
            self.0
                .post(format!("{PREFIX}/stop"))
                .json(&json!({"mode": mode})),
        )?;
        if let Some(error) = result
            .iter()
            .find_map(|item| item.get("error").filter(|value| !value.is_null()))
        {
            return Err(format!("Studio could not stop all runs: {error}").into());
        }
        Ok(())
    }

    pub fn window(&self, project: Option<&str>, ready: bool) -> service::Result<WindowIntents> {
        Self::send(self.0.post(format!("{PREFIX}/window")).json(&json!({
                "pid": std::process::id(), "project": project, "ready": ready,
        })))
    }

    pub fn acknowledge_documents(&self, generation: &str, through: u64) -> service::Result<()> {
        Self::send::<Value>(self.0.post(format!("{PREFIX}/ack-documents")).json(&json!({
            "generation": generation, "through": through,
        })))?;
        Ok(())
    }

    pub fn acknowledge(&self, intent: u64) -> service::Result<()> {
        Self::send::<Value>(self.0.post(format!("{PREFIX}/ack/{intent}")))?;
        Ok(())
    }
}

#[derive(Deserialize)]
struct Snapshot {
    cursor: Option<u64>,
    after: Option<u64>,
    count: usize,
    terminal: Option<Terminal>,
    window: Option<Window>,
}

#[derive(Deserialize)]
struct Window {
    pid: i32,
    project: Option<String>,
}

#[derive(Clone, Deserialize)]
struct Terminal {
    id: String,
    project: String,
    name: String,
    status: TerminalStatus,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TerminalStatus {
    Succeeded,
    Failed,
}

impl TerminalStatus {
    fn label(&self) -> &'static str {
        match self {
            Self::Succeeded => "Done",
            Self::Failed => "Failed",
        }
    }
}

fn should_notify(project: &str, window: Option<&Window>, frontmost: Option<i32>) -> bool {
    !window.is_some_and(|window| {
        Some(window.pid) == frontmost && window.project.as_deref() == Some(project)
    })
}

#[derive(Clone, Default, Deserialize, Serialize)]
struct NotificationCursor {
    cursor: Option<u64>,
    pending: Option<String>,
}

enum Command {
    Open { activity: bool },
    OpenDocuments(Vec<tauri::Url>),
    Stop { force: bool },
    Quit,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[name = "FeedbaxBackgroundActivityActions"]
    #[thread_kind = MainThreadOnly]
    #[ivars = Sender<Command>]
    struct Actions;

    impl Actions {
        #[unsafe(method(openActivity:))]
        fn open(&self, _: &NSMenuItem) {
            self.send(Command::Open { activity: true });
        }

        #[unsafe(method(stopRuns:))]
        fn stop(&self, item: &NSMenuItem) {
            self.send(Command::Stop { force: item.isAlternate() });
        }

        #[unsafe(method(quitService:))]
        fn quit(&self, _: &NSMenuItem) {
            self.send(Command::Quit);
        }
    }

    unsafe impl NSObjectProtocol for Actions {}

    #[allow(non_snake_case)]
    unsafe impl NSApplicationDelegate for Actions {
        #[unsafe(method(applicationShouldHandleReopen:hasVisibleWindows:))]
        fn applicationShouldHandleReopen_hasVisibleWindows(&self, _: &NSApplication, _: bool) -> bool {
            self.send(Command::Open { activity: false });
            false
        }

        #[unsafe(method(application:openURLs:))]
        fn application_openURLs(&self, _: &NSApplication, urls: &NSArray<NSURL>) {
            let urls: Vec<_> = urls
                .iter()
                .filter_map(|url| url.absoluteString()?.to_string().parse().ok())
                .filter(|url| open_path(url).is_some())
                .collect();
            if !urls.is_empty() {
                self.send(Command::OpenDocuments(urls));
            }
        }
    }
);

impl Actions {
    fn send(&self, command: Command) {
        if self.ivars().send(command).is_err() {
            eprintln!("Studio background Activity has stopped");
        }
    }
}

enum Update {
    Snapshot(Snapshot),
    Open {
        pid: Option<i32>,
        activation: File,
        argument: Option<&'static str>,
    },
}

fn save_cursor(
    path: &Path,
    state: &Mutex<NotificationCursor>,
    updated: NotificationCursor,
) -> std::io::Result<()> {
    window_session::write(path, &updated)?;
    *state.lock().unwrap() = updated;
    Ok(())
}

pub struct Menu {
    main: MainThreadMarker,
    item: Retained<NSStatusItem>,
    _actions: Retained<Actions>,
    commands: Receiver<Command>,
    release: service::Release,
    control: Control,
    cursor_path: PathBuf,
    cursor: NotificationCursor,
}

impl Menu {
    pub fn new(release: &service::Release, paths: &service::Paths) -> service::Result<Self> {
        let main =
            MainThreadMarker::new().ok_or("Background Activity must start on the main thread")?;
        let app = NSApplication::sharedApplication(main);
        if !app.setActivationPolicy(NSApplicationActivationPolicy::Accessory) {
            return Err("Could not set the Studio service activation policy".into());
        }
        let control = Control::new(paths)?;
        let cursor_path = paths.control.join("activity-notifications.json");
        let cursor = window_session::read(&cursor_path)?;
        let item = NSStatusBar::systemStatusBar().statusItemWithLength(NSVariableStatusItemLength);
        item.setVisible(false);
        let menu = NSMenu::new(main);
        menu.setAutoenablesItems(false);
        let (sender, commands) = mpsc::channel();
        let actions: Retained<Actions> =
            unsafe { objc2::msg_send![super(main.alloc().set_ivars(sender)), init] };
        app.setDelegate(Some(ProtocolObject::from_ref(&*actions)));
        for (title, selector, alternate) in [
            ("Activity", sel!(openActivity:), false),
            ("Stop All", sel!(stopRuns:), false),
            ("Force Stop All", sel!(stopRuns:), true),
            ("Quit Service", sel!(quitService:), false),
        ] {
            let entry = unsafe {
                NSMenuItem::initWithTitle_action_keyEquivalent(
                    main.alloc(),
                    &NSString::from_str(title),
                    Some(selector),
                    &NSString::from_str(""),
                )
            };
            unsafe { entry.setTarget(Some(&actions)) };
            if alternate {
                entry.setAlternate(true);
                entry.setKeyEquivalentModifierMask(NSEventModifierFlags::Option);
            }
            menu.addItem(&entry);
        }
        item.setMenu(Some(&menu));
        Ok(Self {
            main,
            item,
            _actions: actions,
            commands,
            release: release.clone(),
            control,
            cursor_path,
            cursor,
        })
    }

    pub fn run(self, child: Child) -> service::Result<ExitStatus> {
        let Self {
            main,
            item,
            _actions,
            commands,
            release,
            control,
            cursor_path,
            cursor,
        } = self;
        let app = NSApplication::sharedApplication(main);
        let (updates, incoming) = mpsc::channel();
        let cursor = Arc::new(Mutex::new(cursor));
        let alive = Arc::new(AtomicBool::new(true));
        let worker = {
            let alive = alive.clone();
            let cursor = cursor.clone();
            let release = release.clone();
            thread::spawn(move || monitor(&control, &release, &commands, &updates, &cursor, &alive))
        };
        let result = Rc::new(RefCell::new(None));
        let app_for_timer = app.clone();
        let result_for_timer = result.clone();
        let child = Rc::new(RefCell::new(child));
        let child_for_timer = child.clone();
        let launch = Arc::new(Mutex::new(GuiLaunch::default()));
        let stop_forwarded = Cell::new(false);
        let pending = Arc::new(AtomicBool::new(false));
        let handled = Arc::new(Mutex::new(None));
        let block = RcBlock::new(move |_| {
            if service::background_stop_requested() && !stop_forwarded.replace(true) {
                let pid = child_for_timer.borrow().id() as libc::pid_t;
                unsafe { libc::kill(pid, libc::SIGTERM) };
            }
            let exit = child_for_timer.borrow_mut().try_wait().transpose();
            if let Some(outcome) = exit {
                *result_for_timer.borrow_mut() = Some(outcome);
                app_for_timer.stop(None);
                let wake = NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2(
                        NSEventType::ApplicationDefined, NSPoint::ZERO, NSEventModifierFlags::empty(),
                        0.0, 0, None, 0, 0, 0,
                    );
                if let Some(wake) = wake {
                    app_for_timer.postEvent_atStart(&wake, false);
                }
                return;
            }
            if service::background_stop_requested() {
                return;
            }
            {
                let mut handled = handled.lock().unwrap();
                if let Some(next) = *handled {
                    match save_cursor(
                        &cursor_path,
                        &cursor,
                        NotificationCursor {
                            cursor: Some(next),
                            pending: None,
                        },
                    ) {
                        Ok(()) => *handled = None,
                        Err(error) => {
                            eprintln!("Could not save Studio notification state: {error}")
                        }
                    }
                }
            }
            for update in incoming.try_iter() {
                match update {
                    Update::Snapshot(snapshot) => {
                        item.setVisible(snapshot.count > 0);
                        if let Some(button) = item.button(main) {
                            button.setTitle(&NSString::from_str(&format!(
                                "Feedbax {}",
                                snapshot.count
                            )));
                        }
                        if let Some(next) = snapshot.cursor.filter(|next| {
                            let committed = cursor.lock().unwrap().cursor;
                            snapshot.after == committed
                                && committed != Some(*next)
                                && !pending.load(Ordering::Acquire)
                                && handled.lock().unwrap().is_none()
                        }) {
                            let frontmost = NSWorkspace::sharedWorkspace()
                                .frontmostApplication()
                                .map(|application| application.processIdentifier());
                            let previous = cursor.lock().unwrap().clone();
                            if let Some(terminal) = snapshot.terminal.filter(|terminal| {
                                previous.pending.as_deref() == Some(terminal.id.as_str())
                                    || should_notify(
                                        &terminal.project,
                                        snapshot.window.as_ref(),
                                        frontmost,
                                    )
                            }) {
                                let recovering =
                                    previous.pending.as_deref() == Some(terminal.id.as_str());
                                if !recovering {
                                    if let Err(error) = save_cursor(
                                        &cursor_path,
                                        &cursor,
                                        NotificationCursor {
                                            cursor: previous.cursor,
                                            pending: Some(terminal.id.clone()),
                                        },
                                    ) {
                                        eprintln!("Could not reserve Studio notification delivery: {error}");
                                        continue;
                                    }
                                }
                                pending.store(true, Ordering::Release);
                                let pending = pending.clone();
                                let handled = handled.clone();
                                notify(terminal, recovering, move |delivered| {
                                    if delivered {
                                        *handled.lock().unwrap() = Some(next);
                                    }
                                    pending.store(false, Ordering::Release);
                                });
                            } else {
                                if let Err(error) = save_cursor(
                                    &cursor_path,
                                    &cursor,
                                    NotificationCursor {
                                        cursor: Some(next),
                                        pending: None,
                                    },
                                ) {
                                    eprintln!("Could not save Studio notification state: {error}");
                                }
                            }
                        }
                    }
                    Update::Open {
                        pid,
                        activation,
                        argument,
                    } => {
                        let mut state = launch.lock().unwrap();
                        let starting_pid = state
                            .started
                            .filter(|(_, at)| at.elapsed() < Duration::from_secs(8));
                        let running = pid
                            .into_iter()
                            .chain(starting_pid.map(|(pid, _)| pid))
                            .find_map(|pid| gui_application(&release, pid));
                        if let Some(application) = running {
                            drop(state);
                            application.activateWithOptions(
                                objc2_app_kit::NSApplicationActivationOptions::empty(),
                            );
                        } else if !state.in_flight {
                            state.in_flight = true;
                            drop(state);
                            launch_gui(&release, activation, argument, launch.clone());
                        }
                    }
                }
            }
        });
        let timer = unsafe { NSTimer::timerWithTimeInterval_repeats_block(0.2, true, &block) };
        unsafe {
            NSRunLoop::currentRunLoop()
                .addTimer_forMode(&timer, objc2_foundation::NSRunLoopCommonModes)
        };
        app.run();
        timer.invalidate();
        alive.store(false, Ordering::Relaxed);
        drop(timer);
        drop(block);
        if result.borrow().is_none() {
            let mut child = child.borrow_mut();
            child.kill()?;
            child.wait()?;
        }
        worker
            .join()
            .map_err(|_| "Studio background Activity worker panicked")?;
        match result.take() {
            Some(status) => Ok(status?),
            None => Err("The Studio background event loop stopped before the service".into()),
        }
    }
}

fn monitor(
    control: &Control,
    release: &service::Release,
    commands: &Receiver<Command>,
    updates: &Sender<Update>,
    cursor: &Mutex<NotificationCursor>,
    alive: &AtomicBool,
) {
    let mut quitting: Option<Receiver<service::Result<Value>>> = None;
    let mut last_error = None;
    let mut document_error = None;
    let mut pending_documents = VecDeque::new();
    let mut sequence = 0;
    let paths = match release.paths() {
        Ok(paths) => paths,
        Err(error) => {
            eprintln!("Could not read Studio service paths: {error}");
            return;
        }
    };
    while alive.load(Ordering::Relaxed) {
        let command = commands.recv_timeout(Duration::from_millis(500));
        let operation = match command {
            Ok(Command::Open { activity }) => (|| {
                let activation = release.activity_activation(&paths)?;
                let pid = control.open(activity)?;
                let _ = updates.send(Update::Open {
                    pid,
                    activation,
                    argument: activity.then_some(OPEN_ACTIVITY),
                });
                Ok(())
            })(),
            Ok(Command::OpenDocuments(urls)) => {
                sequence += 1;
                pending_documents.push_back(DocumentOpen { sequence, urls });
                Ok(())
            }
            Ok(Command::Stop { force }) => control.stop(force),
            Ok(Command::Quit) => {
                if quitting.is_none() {
                    quitting = Some(request_quit(release.clone()));
                }
                Ok(())
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(()),
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if let Err(error) = operation {
            eprintln!("Studio background action failed: {error}");
        }
        if let Some(documents) = pending_documents.front() {
            match release.activity_activation(&paths).and_then(|activation| {
                let pid = control.open_documents(documents)?;
                let _ = updates.send(Update::Open {
                    pid,
                    activation,
                    argument: Some(OPEN_DOCUMENTS),
                });
                Ok(())
            }) {
                Ok(()) => {
                    pending_documents.pop_front();
                    document_error = None;
                }
                Err(error) => {
                    let detail = error.to_string();
                    if document_error.as_ref() != Some(&detail) {
                        eprintln!("Could not forward Studio documents: {detail}");
                        document_error = Some(detail);
                    }
                }
            }
        }
        if let Some(reply) = &quitting {
            match reply.try_recv() {
                Ok(Ok(value)) if value["status"] == "timed_out" => {
                    quitting = Some(request_quit(release.clone()));
                }
                Ok(Ok(value)) if value["status"] == "ready_to_stop" => quitting = None,
                Ok(Ok(value)) => {
                    eprintln!("Studio service did not stop: {value}");
                    quitting = None;
                }
                Ok(Err(error)) => {
                    eprintln!("Could not quit the Studio service: {error}");
                    quitting = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => quitting = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let after = cursor.lock().unwrap().cursor;
        match control.snapshot(after) {
            Ok(snapshot) => {
                last_error = None;
                let _ = updates.send(Update::Snapshot(snapshot));
            }
            Err(error) => {
                let error = error.to_string();
                if last_error.as_ref() != Some(&error) {
                    eprintln!("Could not read Studio background Activity: {error}");
                    last_error = Some(error);
                }
            }
        }
    }
}

fn request_quit(release: service::Release) -> Receiver<service::Result<Value>> {
    let (send, receive) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = send.send(release.quit_background());
    });
    receive
}

fn gui_application(release: &service::Release, pid: i32) -> Option<Retained<NSRunningApplication>> {
    NSRunningApplication::runningApplicationWithProcessIdentifier(pid).filter(|application| {
        !application.isTerminated()
            && application
                .bundleURL()
                .and_then(|url| url.path())
                .is_some_and(|path| Path::new(&path.to_string()) == release.bundle_path())
    })
}

#[derive(Default)]
struct GuiLaunch {
    in_flight: bool,
    started: Option<(i32, Instant)>,
}

fn launch_gui(
    release: &service::Release,
    activation: File,
    argument: Option<&str>,
    launch: Arc<Mutex<GuiLaunch>>,
) {
    let url = NSURL::fileURLWithPath_isDirectory(
        &NSString::from_str(&release.bundle_path().to_string_lossy()),
        true,
    );
    let configuration = NSWorkspaceOpenConfiguration::configuration();
    configuration.setCreatesNewApplicationInstance(true);
    configuration.setAllowsRunningApplicationSubstitution(false);
    if let Some(argument) = argument {
        configuration.setArguments(&NSArray::from_retained_slice(&[NSString::from_str(
            argument,
        )]));
    }
    let activation = Mutex::new(Some(activation));
    let callback = RcBlock::new(
        move |application: *mut NSRunningApplication, error: *mut objc2_foundation::NSError| {
            let mut state = launch.lock().unwrap();
            state.started = unsafe { application.as_ref() }
                .map(|app| (app.processIdentifier(), Instant::now()));
            state.in_flight = false;
            drop(activation.lock().unwrap().take());
            drop(state);
            if let Some(error) = unsafe { error.as_ref() } {
                eprintln!("Could not open Studio: {error}");
            }
        },
    );
    NSWorkspace::sharedWorkspace().openApplicationAtURL_configuration_completionHandler(
        &url,
        &configuration,
        Some(&callback),
    );
}

fn notify(terminal: Terminal, recovering: bool, complete: impl Fn(bool) + Send + Sync + 'static) {
    let complete: Arc<dyn Fn(bool) + Send + Sync> = Arc::new(complete);
    if recovering {
        let check = RcBlock::new(
            move |requests: std::ptr::NonNull<objc2_foundation::NSArray<UNNotificationRequest>>| {
                if unsafe { requests.as_ref() }
                    .to_vec()
                    .iter()
                    .any(|request| request.identifier().to_string() == terminal.id)
                {
                    complete(true);
                    return;
                }
                let terminal = terminal.clone();
                let complete = complete.clone();
                let delivered = RcBlock::new(
                    move |notices: std::ptr::NonNull<objc2_foundation::NSArray<UNNotification>>| {
                        if unsafe { notices.as_ref() }
                            .to_vec()
                            .iter()
                            .any(|notice| notice.request().identifier().to_string() == terminal.id)
                        {
                            complete(true);
                        } else {
                            authorized_notification(terminal.clone(), complete.clone());
                        }
                    },
                );
                UNUserNotificationCenter::currentNotificationCenter()
                    .getDeliveredNotificationsWithCompletionHandler(&delivered);
            },
        );
        UNUserNotificationCenter::currentNotificationCenter()
            .getPendingNotificationRequestsWithCompletionHandler(&check);
    } else {
        authorized_notification(terminal, complete);
    }
}

fn authorized_notification(terminal: Terminal, complete: Arc<dyn Fn(bool) + Send + Sync>) {
    let center = UNUserNotificationCenter::currentNotificationCenter();
    let callback = RcBlock::new(
        move |settings: std::ptr::NonNull<objc2_user_notifications::UNNotificationSettings>| {
            let settings = unsafe { settings.as_ref() };
            match settings.authorizationStatus() {
                UNAuthorizationStatus::Authorized | UNAuthorizationStatus::Provisional => {
                    deliver(&terminal, complete.clone());
                }
                UNAuthorizationStatus::NotDetermined => {
                    let terminal = terminal.clone();
                    let complete = complete.clone();
                    let callback = RcBlock::new(
                        move |allowed: objc2::runtime::Bool,
                              error: *mut objc2_foundation::NSError| {
                            if let Some(error) = unsafe { error.as_ref() } {
                                eprintln!("Studio notification authorization failed: {error}");
                                complete(false);
                            } else if allowed.as_bool() {
                                deliver(&terminal, complete.clone());
                            } else {
                                complete(true);
                            }
                        },
                    );
                    UNUserNotificationCenter::currentNotificationCenter()
                        .requestAuthorizationWithOptions_completionHandler(
                            UNAuthorizationOptions::Alert | UNAuthorizationOptions::Provisional,
                            &callback,
                        );
                }
                _ => complete(true),
            }
        },
    );
    center.getNotificationSettingsWithCompletionHandler(&callback);
}

fn deliver(terminal: &Terminal, complete: Arc<dyn Fn(bool) + Send + Sync>) {
    let content = UNMutableNotificationContent::new();
    content.setTitle(&NSString::from_str(&terminal.name));
    content.setBody(&NSString::from_str(terminal.status.label()));
    let request = UNNotificationRequest::requestWithIdentifier_content_trigger(
        &NSString::from_str(&terminal.id),
        &content,
        None,
    );
    let callback = RcBlock::new(move |error: *mut objc2_foundation::NSError| {
        if let Some(error) = unsafe { error.as_ref() } {
            eprintln!("Studio notification could not be delivered: {error}");
        }
        complete(error.is_null());
    });
    UNUserNotificationCenter::currentNotificationCenter()
        .addNotificationRequest_withCompletionHandler(&request, Some(&callback));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_notification_is_suppressed_only_for_its_foreground_project() {
        let window = Window {
            pid: 11,
            project: Some("reach".into()),
        };
        assert!(!should_notify("reach", Some(&window), Some(11)));
        assert!(should_notify("other", Some(&window), Some(11)));
        assert!(should_notify("reach", Some(&window), Some(12)));
        assert!(should_notify("reach", None, Some(11)));
    }
}

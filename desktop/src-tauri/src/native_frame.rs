use block2::RcBlock;
use objc2::{rc::Retained, runtime::AnyObject, MainThreadMarker, Message};
use objc2_app_kit::NSWindow;
use objc2_foundation::{NSError, NSString};
use objc2_web_kit::{WKContentWorld, WKWebView};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tauri::{Manager, WebviewWindow, WindowEvent};
use window_shell_geometry::{
    appkit::{observe_controls, ControlsObserver},
    host_identity_script, window_controls_script, PlacementError, WindowControls,
};

thread_local! {
    static FRAMES: RefCell<BTreeMap<u64, Hosted>> = const { RefCell::new(BTreeMap::new()) };
}
static NEXT_WINDOW: AtomicU64 = AtomicU64::new(0);

struct Hosted {
    observer: ControlsObserver,
    webview: Retained<WKWebView>,
}

pub fn flush_preferences(
    window: &WebviewWindow,
    completed: impl Fn(Result<(), String>) + Send + 'static,
) -> tauri::Result<()> {
    window.with_webview(move |platform| {
        let Some(main) = MainThreadMarker::new() else {
            completed(Err("Window preferences require the main thread.".into()));
            return;
        };
        let webview = unsafe { &*platform.inner().cast::<WKWebView>() };
        let callback = RcBlock::new(move |_: *mut AnyObject, error: *mut NSError| {
            completed(match unsafe { error.as_ref() } {
                Some(error) => Err(error.localizedDescription().to_string()),
                None => Ok(()),
            });
        });
        let script = NSString::from_str("await window.__FEEDBAX_FLUSH_WINDOW_PREFERENCES__?.();");
        unsafe {
            webview.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler(
                &script,
                None,
                None,
                &WKContentWorld::pageWorld(main),
                Some(&callback),
            );
        }
    })
}

struct FrameState {
    origin: String,
    generation: u64,
}

/// The native side of a Studio window's title bar: AppKit places the window controls and the page learns where they are.
#[derive(Clone)]
pub struct StudioFrame(Arc<FrameState>);

impl StudioFrame {
    pub fn new(url: &tauri::Url) -> tauri::Result<Self> {
        if url.scheme() != "http" || url.host_str() != Some("127.0.0.1") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Studio frame requires the local service origin",
            )
            .into());
        }
        Ok(Self(Arc::new(FrameState {
            origin: url.origin().ascii_serialization(),
            generation: NEXT_WINDOW.fetch_add(1, Ordering::Relaxed),
        })))
    }

    pub fn matches(&self, url: &tauri::Url) -> bool {
        url.origin().ascii_serialization() == self.0.origin
    }

    fn scoped(&self, script: &str) -> String {
        let origin = serde_json::json!(self.0.origin);
        format!("if (location.origin === {origin}) {{ {script} }}")
    }

    /// Marks documents from the Studio origin as natively hosted before their own scripts run.
    pub fn host_script(&self) -> String {
        self.scoped(&host_identity_script())
    }

    fn deliver(&self, webview: &WKWebView, outcome: Result<WindowControls, PlacementError>) {
        let controls = match outcome {
            Ok(controls) => controls,
            Err(error) => {
                eprintln!("Studio window controls could not be placed: {error}");
                return;
            }
        };
        let script = NSString::from_str(&self.scoped(&window_controls_script(controls)));
        let completion = RcBlock::new(|_: *mut AnyObject, error: *mut NSError| {
            if let Some(error) = unsafe { error.as_ref() } {
                eprintln!(
                    "Studio window controls could not be reported to the page: {}",
                    error.localizedDescription()
                );
            }
        });
        unsafe { webview.evaluateJavaScript_completionHandler(&script, Some(&completion)) };
    }

    /// Gives a newly committed document the controls' current placement, which reports before it existed missed.
    pub fn page_committed(&self) {
        MainThreadMarker::new().expect("Wry reports page loads on the main thread");
        FRAMES.with(|frames| {
            if let Some(hosted) = frames.borrow().get(&self.0.generation) {
                self.deliver(&hosted.webview, hosted.observer.current());
            }
        });
    }

    pub fn install(&self, window: &WebviewWindow) -> tauri::Result<()> {
        let target = window.clone();
        let generation = self.0.generation;
        window.on_window_event(move |event| {
            if let WindowEvent::Destroyed = event {
                if let Err(error) = target.app_handle().run_on_main_thread(move || {
                    FRAMES.with(|frames| frames.borrow_mut().remove(&generation));
                }) {
                    eprintln!("Studio window controls observer was not released: {error}");
                }
            }
        });

        let frame = self.clone();
        window.with_webview(move |platform| {
            let main = MainThreadMarker::new().expect("Wry runs with_webview on the main thread");
            let native = unsafe { &*platform.ns_window().cast::<NSWindow>() };
            let webview = unsafe { &*platform.inner().cast::<WKWebView>() };
            let reporting = webview.retain();
            let reporter = frame.clone();
            let observer = observe_controls(main, native, webview, move |outcome| {
                reporter.deliver(&reporting, outcome)
            });
            FRAMES.with(|frames| {
                frames.borrow_mut().insert(
                    frame.0.generation,
                    Hosted {
                        observer,
                        webview: webview.retain(),
                    },
                );
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_scoped_to_the_studio_service_origin() {
        let frame = StudioFrame::new(&"http://127.0.0.1:41000/".parse().unwrap()).unwrap();
        assert!(frame.matches(&"http://127.0.0.1:41000/projects/first".parse().unwrap()));
        assert!(StudioFrame::new(&"https://example.test/".parse().unwrap()).is_err());
        for other in [
            "http://127.0.0.1:41001/",
            "https://127.0.0.1:41000/",
            "http://localhost:41000/",
            "tauri://localhost/",
        ] {
            assert!(!frame.matches(&other.parse().unwrap()));
        }
    }
}

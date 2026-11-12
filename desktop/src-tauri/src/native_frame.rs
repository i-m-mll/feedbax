use block2::RcBlock;
use objc2::{
    rc::Retained,
    runtime::{AnyObject, NSObjectProtocol, ProtocolObject},
};
use objc2_app_kit::{
    NSWindow, NSWindowDidEnterFullScreenNotification, NSWindowDidExitFullScreenNotification,
};
use objc2_foundation::{NSError, NSNotification, NSNotificationCenter, NSOperationQueue, NSString};
use objc2_web_kit::WKWebView;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};
use tauri::{Manager, WebviewWindow, WindowEvent};
use window_shell_geometry::appkit::overlay_reservation;

struct Observations {
    center: Retained<NSNotificationCenter>,
    tokens: Vec<Retained<ProtocolObject<dyn NSObjectProtocol>>>,
}

impl Drop for Observations {
    fn drop(&mut self) {
        for token in &self.tokens {
            unsafe { self.center.removeObserver((&**token).as_ref()) };
        }
    }
}

thread_local! {
    static OBSERVATIONS: RefCell<BTreeMap<u64, Observations>> = RefCell::new(BTreeMap::new());
}
static NEXT_WINDOW: AtomicU64 = AtomicU64::new(0);

struct FrameState {
    origin: String,
    live: AtomicBool,
    refresh_queued: AtomicBool,
    reported_failure: AtomicBool,
}

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
            live: AtomicBool::new(true),
            refresh_queued: AtomicBool::new(false),
            reported_failure: AtomicBool::new(false),
        })))
    }

    pub fn matches(&self, url: &tauri::Url) -> bool {
        url.origin().ascii_serialization() == self.0.origin
    }

    fn report(&self, error: impl std::fmt::Display) {
        if !self.0.reported_failure.swap(true, Ordering::Relaxed) {
            eprintln!("Studio window controls could not be measured: {error}");
        }
    }

    pub fn refresh(&self, window: &WebviewWindow) {
        if !self.0.live.load(Ordering::Acquire)
            || self.0.refresh_queued.swap(true, Ordering::AcqRel)
        {
            return;
        }
        let target = window.clone();
        let frame = self.clone();
        let refresh = RcBlock::new(move || {
            frame.0.refresh_queued.store(false, Ordering::Release);
            frame.measure(&target);
        });
        // Wry dispatches with_webview synchronously on the main thread.
        unsafe { NSOperationQueue::mainQueue().addOperationWithBlock(&refresh) };
    }

    fn measure(&self, window: &WebviewWindow) {
        if !self.0.live.load(Ordering::Acquire) {
            return;
        }
        let frame = self.clone();
        if let Err(error) = window.with_webview(move |platform| {
            if !frame.0.live.load(Ordering::Acquire) {
                return;
            }
            let native = unsafe { &*platform.ns_window().cast::<NSWindow>() };
            let view = unsafe { &*platform.inner().cast::<WKWebView>() };
            match overlay_reservation(native, view, unsafe { view.pageZoom() }) {
                Ok(value) if value.is_finite() => {
                    let script = NSString::from_str(&frame.update_script(value.max(0.0)));
                    let completion = RcBlock::new(move |_: *mut AnyObject, error: *mut NSError| {
                        if let Some(error) = unsafe { error.as_ref() } {
                            frame.report(error.localizedDescription());
                        }
                    });
                    unsafe {
                        view.evaluateJavaScript_completionHandler(&script, Some(&completion))
                    };
                }
                Ok(_) => frame.report("Invalid CSS reservation"),
                Err(error) => frame.report(error),
            }
        }) {
            self.report(error);
        }
    }

    fn update_script(&self, reservation: f64) -> String {
        let origin = serde_json::json!(self.0.origin);
        format!(
            "if (location.origin === {origin}) {{\
               const apply = () => document.documentElement.style.setProperty(\
                 '--studio-native-controls', '{reservation}px');\
               if (document.documentElement) apply();\
               else addEventListener('DOMContentLoaded', apply, {{once: true}});\
             }}"
        )
    }

    pub fn install(&self, window: &WebviewWindow) -> tauri::Result<()> {
        let generation = NEXT_WINDOW.fetch_add(1, Ordering::Relaxed);
        let target = window.clone();
        let frame = self.clone();
        window.on_window_event(move |event| match event {
            WindowEvent::Destroyed => {
                frame.0.live.store(false, Ordering::Release);
                if let Err(error) = target.app_handle().run_on_main_thread(move || {
                    OBSERVATIONS.with(|observations| {
                        observations.borrow_mut().remove(&generation);
                    });
                }) {
                    frame.report(error);
                }
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                frame.refresh(&target);
            }
            _ => {}
        });

        let target = window.clone();
        let frame = self.clone();
        window.with_webview(move |platform| {
            if !frame.0.live.load(Ordering::Acquire) {
                return;
            }
            let native = unsafe { &*platform.ns_window().cast::<NSWindow>() };
            let center = NSNotificationCenter::defaultCenter();
            let tokens = [unsafe { NSWindowDidEnterFullScreenNotification }, unsafe {
                NSWindowDidExitFullScreenNotification
            }]
            .into_iter()
            .map(|name| {
                let target = target.clone();
                let frame = frame.clone();
                let callback = RcBlock::new(move |_: NonNull<NSNotification>| {
                    frame.refresh(&target);
                });
                unsafe {
                    center.addObserverForName_object_queue_usingBlock(
                        Some(name),
                        Some(native),
                        None,
                        &callback,
                    )
                }
            })
            .collect();
            OBSERVATIONS.with(|observations| {
                observations
                    .borrow_mut()
                    .insert(generation, Observations { center, tokens });
            });
        })?;
        self.refresh(window);
        Ok(())
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

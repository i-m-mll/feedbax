use crate::service;
use reqwest::blocking::{Client, RequestBuilder};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use std::time::Duration;

const PREFIX: &str = "http://localhost/_desktop/activity";
pub const OPEN_ACTIVITY: &str = "--activity";

#[derive(Deserialize)]
struct GuiTarget {
    pid: Option<i32>,
}

#[derive(Deserialize)]
struct WindowIntent {
    open: Option<u64>,
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

    pub(crate) fn snapshot(&self, after: Option<u64>) -> service::Result<Snapshot> {
        let url = after.map_or_else(
            || PREFIX.to_owned(),
            |cursor| format!("{PREFIX}?after={cursor}"),
        );
        Self::send(self.0.get(url))
    }

    pub(crate) fn open(&self) -> service::Result<Option<i32>> {
        Ok(Self::send::<GuiTarget>(self.0.post(format!("{PREFIX}/open")))?.pid)
    }

    pub(crate) fn stop(&self, force: bool) -> service::Result<()> {
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

    pub fn window(&self, project: Option<&str>, ready: bool) -> service::Result<Option<u64>> {
        let intent: WindowIntent =
            Self::send(self.0.post(format!("{PREFIX}/window")).json(&json!({
                "pid": std::process::id(), "project": project, "ready": ready,
            })))?;
        Ok(intent.open)
    }

    pub fn acknowledge(&self, intent: u64) -> service::Result<()> {
        Self::send::<Value>(self.0.post(format!("{PREFIX}/ack/{intent}")))?;
        Ok(())
    }
}

#[derive(Deserialize)]
pub(crate) struct Snapshot {
    pub(crate) cursor: Option<u64>,
    pub(crate) after: Option<u64>,
    pub(crate) count: usize,
    pub(crate) terminal: Option<Terminal>,
    pub(crate) window: Option<Window>,
}

#[derive(Deserialize)]
pub(crate) struct Window {
    pub(crate) pid: i32,
    pub(crate) project: Option<String>,
}

#[derive(Clone, Deserialize)]
pub(crate) struct Terminal {
    pub(crate) id: String,
    pub(crate) project: String,
    pub(crate) name: String,
    pub(crate) status: TerminalStatus,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TerminalStatus {
    Succeeded,
    Failed,
}

impl TerminalStatus {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Succeeded => "Done",
            Self::Failed => "Failed",
        }
    }
}

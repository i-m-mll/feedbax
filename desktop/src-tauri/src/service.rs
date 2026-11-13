use block2::RcBlock;
use objc2_foundation::{NSBundle, NSError, NSString};
use objc2_service_management::{SMAppService, SMAppServiceStatus};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::fd::{AsRawFd, FromRawFd, RawFd},
    os::unix::fs::MetadataExt,
    path::{Component, Path, PathBuf},
    process::{self, Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Deserialize)]
struct Manifest {
    schema: String,
    service_protocol: u32,
    bundle_id: String,
    resources: BTreeMap<String, String>,
}

pub struct Release {
    contents: PathBuf,
    bundle: String,
    python: PathBuf,
    installer: PathBuf,
}

#[derive(Deserialize)]
pub struct Paths {
    pub logs: PathBuf,
    pub control: PathBuf,
    pub projects: PathBuf,
    pub settings: PathBuf,
    pub storage_id: [u8; 16],
}

impl Release {
    pub fn load() -> Result<Self> {
        let bundle = NSBundle::mainBundle();
        let contents = PathBuf::from(bundle.bundlePath().to_string())
            .join("Contents")
            .canonicalize()?;
        let identifier = bundle
            .bundleIdentifier()
            .ok_or("Studio must launch from its installed app bundle")?
            .to_string();
        let manifest: Manifest =
            serde_json::from_slice(&fs::read(contents.join("Resources/release.json"))?)?;
        if manifest.schema != "feedbax.desktop.release.v1" || manifest.service_protocol != 1 {
            return Err("Unsupported Studio release manifest or service protocol".into());
        }
        if identifier != manifest.bundle_id {
            return Err(
                "Installed application and release manifest have different bundle identities"
                    .into(),
            );
        }
        let python = installed_resource(&contents, &manifest, "python")?;
        let installer = installed_resource(&contents, &manifest, "installer")?;
        Ok(Self {
            contents,
            bundle: identifier,
            python,
            installer,
        })
    }

    pub fn run_installer(&self, args: &[String]) -> Result<()> {
        let status = Command::new(&self.python)
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .args(["-I", "-B", "-c",
                "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('install', run_name='__main__')"])
            .arg(&self.installer)
            .args(args)
            .status()?;
        if !status.success() {
            return Err(format!("Studio installer exited with {status}").into());
        }
        Ok(())
    }

    fn command(&self, action: &str) -> Command {
        let mut command = Command::new(&self.python);
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .args(["-I", "-B", "-m", "feedbax.desktop", "--contents"])
            .arg(&self.contents)
            .arg(action);
        command
    }

    pub fn call(&self, action: &str, args: &[&str]) -> Result<Value> {
        let output = self.command(action).args(args).output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Studio {action} failed: {}", stderr.trim()).into());
        }
        Ok(serde_json::from_slice(&output.stdout)?)
    }

    pub fn paths(&self) -> Result<Paths> {
        Ok(serde_json::from_value(self.call("paths", &[])?)?)
    }

    fn registration(&self) -> objc2::rc::Retained<SMAppService> {
        let plist = NSString::from_str(&format!("{}.runtime.plist", self.bundle));
        unsafe { SMAppService::agentServiceWithPlistName(&plist) }
    }

    pub fn registration_status(&self) -> &'static str {
        status_name(unsafe { self.registration().status() })
    }

    pub fn register(&self) -> Result<&'static str> {
        let agent = self.registration();
        unsafe {
            if matches!(
                agent.status(),
                SMAppServiceStatus::NotRegistered | SMAppServiceStatus::NotFound
            ) {
                if let Err(error) = agent.registerAndReturnError() {
                    let status = agent.status();
                    if !matches!(
                        status,
                        SMAppServiceStatus::Enabled | SMAppServiceStatus::RequiresApproval
                    ) {
                        return Err(format!(
                            "Could not register the Studio background service: {error}"
                        )
                        .into());
                    }
                }
            }
            Ok(status_name(agent.status()))
        }
    }

    pub fn unregister(&self) -> Result<&'static str> {
        let agent = self.registration();
        unsafe {
            if matches!(
                agent.status(),
                SMAppServiceStatus::Enabled | SMAppServiceStatus::RequiresApproval
            ) {
                let (tx, rx) = mpsc::sync_channel(1);
                let completed = RcBlock::new(move |error: *mut NSError| {
                    let result = if error.is_null() {
                        Ok(())
                    } else {
                        Err((&*error).to_string())
                    };
                    let _ = tx.send(result);
                });
                agent.unregisterWithCompletionHandler(&completed);
                rx.recv_timeout(Duration::from_secs(10))??;
            }
            let status = agent.status();
            if !matches!(
                status,
                SMAppServiceStatus::NotRegistered | SMAppServiceStatus::NotFound
            ) {
                return Err(format!(
                    "Background registration still present ({})",
                    status_name(status)
                )
                .into());
            }
            self.await_agent_absent()?;
            Ok(status_name(status))
        }
    }

    fn await_agent_absent(&self) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(8);
        let label = format!("gui/{}/{}.runtime", unsafe { libc::getuid() }, self.bundle);
        loop {
            let output = Command::new("/bin/launchctl")
                .args(["print", &label])
                .output()?;
            if !output.status.success() {
                let detail = String::from_utf8_lossy(&output.stderr);
                if detail.contains("Could not find service") {
                    return Ok(());
                }
                return Err(
                    format!("Could not confirm Studio agent absence: {}", detail.trim()).into(),
                );
            }
            if Instant::now() >= deadline {
                return Err("Studio agent remains loaded after unregistration".into());
            }
            thread::sleep(Duration::from_millis(125));
        }
    }

    pub fn connect(&self, ticket_args: &[&str]) -> Result<Value> {
        let paths = self.paths()?;
        let Some(_activation) = activation_lock(&paths)? else {
            return Ok(json!({"status": "installing"}));
        };
        if paths.control.join("installation.json").exists() {
            return Ok(json!({"status": "recovery_required"}));
        }
        let health = self.call("health", &[])?;
        match health["status"].as_str() {
            Some("ready") => return self.call("ticket", ticket_args),
            Some("unavailable") => {}
            _ => return Ok(health),
        }
        let status = self.register()?;
        if status != "enabled" {
            return Ok(json!({"status": status}));
        }
        let mut wait = vec!["--wait", "20"];
        wait.extend_from_slice(ticket_args);
        self.call("ticket", &wait)
    }

    pub fn stop(&self, disposition: &str) -> Result<Value> {
        let paths = self.paths()?;
        let Some(_activation) = activation_lock(&paths)? else {
            return Ok(json!({"status": "installing"}));
        };
        if paths.control.join("installation.json").exists() {
            return Ok(json!({"status": "recovery_required"}));
        }
        self.stop_locked(disposition)
    }

    fn stop_locked(&self, disposition: &str) -> Result<Value> {
        let result = self.call("stop", &["--disposition", disposition])?;
        if result["status"] == "ready_to_stop" {
            let deadline = Instant::now() + Duration::from_secs(8);
            loop {
                if self.call("health", &[])?["status"] == "unavailable" {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(
                        "Studio settled its jobs but did not exit before the deadline".into(),
                    );
                }
                thread::sleep(Duration::from_millis(125));
            }
            self.unregister()?;
        }
        Ok(result)
    }

    fn serve(&self) -> Result<()> {
        let logs = self.paths()?.logs;
        fs::create_dir_all(&logs)?;
        let mut child = self
            .command("serve")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().ok_or("Missing Studio stdout pipe")?;
        let stderr = child.stderr.take().ok_or("Missing Studio stderr pipe")?;
        let output_log = logs.join("studio.stdout.log");
        let error_log = logs.join("studio.stderr.log");
        let (status, output, error) = thread::scope(|scope| {
            let output = scope.spawn(move || append_log(stdout, &output_log));
            let error = scope.spawn(move || append_log(stderr, &error_log));
            (child.wait(), output.join(), error.join())
        });
        output.map_err(|_| "Studio stdout logger panicked")??;
        error.map_err(|_| "Studio stderr logger panicked")??;
        let status = status?;
        if !status.success() {
            return Err(format!("Bundled Studio runtime exited with {status}").into());
        }
        Ok(())
    }
}

fn installed_resource(contents: &Path, manifest: &Manifest, role: &str) -> Result<PathBuf> {
    let relative = Path::new(
        manifest
            .resources
            .get(role)
            .ok_or_else(|| format!("Studio release has no {role} resource"))?,
    );
    if !relative
        .components()
        .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(format!("Installed {role} resource must be bundle-relative").into());
    }
    let resource = contents.join(relative).canonicalize()?;
    if !resource.starts_with(contents) {
        return Err(format!("Installed {role} resource is outside the app bundle").into());
    }
    Ok(resource)
}

fn activation_lock(paths: &Paths) -> Result<Option<File>> {
    use std::os::unix::fs::OpenOptionsExt;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(paths.control.join("install.lock"))?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(Some(lock));
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::WouldBlock {
        Ok(None)
    } else {
        Err(error.into())
    }
}

fn mutation_lock(paths: &Paths, descriptor: Option<RawFd>) -> Result<File> {
    let Some(descriptor) = descriptor else {
        return activation_lock(paths)?
            .ok_or_else(|| "A Studio installation is in progress".into());
    };
    let fd = unsafe { libc::fcntl(descriptor, libc::F_DUPFD_CLOEXEC, 3) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let lock = unsafe { File::from_raw_fd(fd) };
    let current = fs::symlink_metadata(paths.control.join("install.lock"))?;
    let inherited = lock.metadata()?;
    if !current.is_file() || current.dev() != inherited.dev() || current.ino() != inherited.ino() {
        return Err("Installer lock does not identify this Studio instance".into());
    }
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(lock)
}

fn status_name(status: SMAppServiceStatus) -> &'static str {
    match status {
        SMAppServiceStatus::NotRegistered => "not_registered",
        SMAppServiceStatus::Enabled => "enabled",
        SMAppServiceStatus::RequiresApproval => "requires_approval",
        SMAppServiceStatus::NotFound => "not_found",
        _ => "unknown_registration",
    }
}

pub fn open_login_items() {
    unsafe { SMAppService::openSystemSettingsLoginItems() };
}

fn append_log(mut stream: impl Read, path: &Path) -> std::io::Result<()> {
    const LIMIT: u64 = 1024 * 1024;
    let mut file = open_log(path)?;
    let mut length = file.metadata()?.len();
    let mut buffer = [0; 8192];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Ok(());
        }
        if length + count as u64 > LIMIT {
            fs::rename(path, path.with_extension("log.1"))?;
            file = open_log(path)?;
            length = 0;
        }
        file.write_all(&buffer[..count])?;
        length += count as u64;
    }
}

fn open_log(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)
}

pub fn cli(args: &[String]) -> Result<()> {
    let release = Release::load()?;
    let action = args.first().ok_or("Missing Studio service operation")?;
    let mut rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
    let descriptor = if rest.len() >= 2 && rest[rest.len() - 2] == "--installation-fd" {
        let descriptor = rest.pop().unwrap().parse::<RawFd>()?;
        rest.pop();
        Some(descriptor)
    } else {
        None
    };
    let answer = match (action.as_str(), rest.as_slice()) {
        ("serve", []) => return release.serve(),
        ("register", []) => {
            let _lock = mutation_lock(&release.paths()?, descriptor)?;
            json!({"status": release.register()?})
        }
        ("unregister", []) => {
            let _lock = mutation_lock(&release.paths()?, descriptor)?;
            if release.call("health", &[])?["status"] != "unavailable" {
                return Err("Stop the active Studio service before unregistering it".into());
            }
            json!({"status": release.unregister()?})
        }
        ("registration", []) => json!({"status": release.registration_status()}),
        ("health", _) | ("paths", []) => release.call(action, &rest)?,
        ("stop", [disposition @ ("defer" | "checkpoint" | "force")]) => {
            let _lock = mutation_lock(&release.paths()?, descriptor)?;
            release.stop_locked(disposition)?
        }
        _ => return Err("Unknown Studio service operation".into()),
    };
    println!("{answer}");
    Ok(())
}

pub fn fail(error: impl std::fmt::Display) -> ! {
    eprintln!("{error}");
    process::exit(1);
}

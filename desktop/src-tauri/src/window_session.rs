use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAX_BYTES: u64 = 1024 * 1024;
static NEXT_WRITE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Folder {
    pub path: PathBuf,
    pub identity: [u64; 2],
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

impl Size {
    pub const MIN: Self = Self {
        width: 820.,
        height: 560.,
    };
    pub const DEFAULT: Self = Self {
        width: 1350.,
        height: 910.,
    };

    pub fn bounded(self, available: Self) -> Self {
        Self {
            width: self
                .width
                .clamp(Self::MIN.width, available.width.max(Self::MIN.width)),
            height: self
                .height
                .clamp(Self::MIN.height, available.height.max(Self::MIN.height)),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SavedWindow {
    pub folder: Folder,
    pub size: Size,
}

pub type OpenWindows = BTreeMap<String, SavedWindow>;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecentProject {
    pub id: String,
    pub title: String,
    pub folder: Folder,
}

#[derive(Clone, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RecentProjects(Vec<RecentProject>);

impl RecentProjects {
    pub const LIMIT: usize = 10;

    pub fn entries(&self) -> &[RecentProject] {
        &self.0
    }

    pub fn record(&mut self, project: RecentProject) {
        self.0
            .retain(|entry| entry.id != project.id && entry.folder.path != project.folder.path);
        self.0.insert(0, project);
        self.0.truncate(Self::LIMIT);
    }

    pub fn rename(&mut self, id: &str, title: &str) -> bool {
        let Some(entry) = self
            .0
            .iter_mut()
            .find(|entry| entry.id == id && entry.title != title)
        else {
            return false;
        };
        entry.title = title.to_owned();
        true
    }
}

pub fn read<T: Default + DeserializeOwned>(path: &Path) -> io::Result<T> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(T::default()),
        Err(error) => return Err(error),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.uid() != unsafe { libc::getuid() }
        || metadata.mode() & 0o077 != 0
        || metadata.len() > MAX_BYTES
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Studio window session file",
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Studio window session is too large",
        ));
    }
    serde_json::from_slice(&bytes).map_err(io::Error::other)
}

pub fn write(path: &Path, state: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec(state).map_err(io::Error::other)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Studio window session is too large",
        ));
    }
    let directory = path.parent().expect("absolute window session path");
    let sequence = NEXT_WRITE.fetch_add(1, Ordering::Relaxed);
    let temporary = directory.join(format!(".studio-windows-{}-{sequence}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)?;
    let result = (|| {
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(directory)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_window_sizes_round_trip_and_close_removes_only_one() {
        let directory =
            std::env::temp_dir().join(format!("feedbax-window-session-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("windows.json");
        let mut windows: OpenWindows = read(&path).unwrap();
        assert!(windows.is_empty());
        for (index, width) in [(1, 1100.), (2, 980.)] {
            windows.insert(
                format!("project-{index}"),
                SavedWindow {
                    folder: Folder {
                        path: directory.join(index.to_string()),
                        identity: [7, index],
                    },
                    size: Size {
                        width,
                        height: 670.,
                    },
                },
            );
        }
        write(&path, &windows).unwrap();
        assert_eq!(read::<OpenWindows>(&path).unwrap(), windows);
        windows.remove("project-2");
        write(&path, &windows).unwrap();
        assert_eq!(read::<OpenWindows>(&path).unwrap(), windows);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn recents_are_bounded_deduplicated_and_independent_of_window_restore() {
        let directory = std::env::temp_dir().join(format!("feedbax-recent-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let windows_path = directory.join("windows.json");
        let recent_path = directory.join("recent.json");
        let mut projects = RecentProjects::default();
        let project = |number: u64| RecentProject {
            id: format!("p-{number}"),
            title: format!("Name-{number}"),
            folder: Folder {
                path: directory.join(number.to_string()),
                identity: [3, number],
            },
        };
        for number in 1..=11 {
            projects.record(project(number));
        }
        assert_eq!(projects.entries().len(), RecentProjects::LIMIT);
        assert_eq!(projects.entries().last().unwrap().id, "p-2");
        projects.record(project(2));
        assert!(projects.rename("p-11", "Renamed"));
        assert_eq!(projects.entries()[0].id, "p-2");
        assert_eq!(projects.entries()[1].title, "Renamed");
        write(&recent_path, &projects).unwrap();
        let restored: RecentProjects = read(&recent_path).unwrap();
        assert!(restored == projects);
        let windows = OpenWindows::from([(
            "p-2".into(),
            SavedWindow {
                folder: project(2).folder,
                size: Size::DEFAULT,
            },
        )]);
        write(&windows_path, &windows).unwrap();
        let mut cleared = RecentProjects::default();
        assert!(!cleared.rename("p-2", "New-name"));
        write(&recent_path, &cleared).unwrap();
        assert!(read::<RecentProjects>(&recent_path)
            .unwrap()
            .entries()
            .is_empty());
        assert_eq!(read::<OpenWindows>(&windows_path).unwrap(), windows);
        fs::remove_dir_all(directory).unwrap();
    }
}

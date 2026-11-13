use serde::{Deserialize, Serialize};
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

pub fn read(path: &Path) -> io::Result<OpenWindows> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
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

pub fn write(path: &Path, windows: &OpenWindows) -> io::Result<()> {
    let bytes = serde_json::to_vec(windows).map_err(io::Error::other)?;
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
        let mut windows = read(&path).unwrap();
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
        assert_eq!(read(&path).unwrap(), windows);
        windows.remove("project-2");
        write(&path, &windows).unwrap();
        assert_eq!(read(&path).unwrap(), windows);
        fs::remove_dir_all(directory).unwrap();
    }
}

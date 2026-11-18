import { runPythonProcess } from './python';

function processId(text: string, owner: string): number {
  const pid = Number(text.trim());
  if (!Number.isSafeInteger(pid) || pid <= 1) throw new Error(`Invalid isolated ${owner} PID`);
  return pid;
}

export async function hostPid(home: string): Promise<number> {
  // The ping starts a Host when none answers, which must not become a child of the worker's
  // shared fixture runner.
  const stdout = await runPythonProcess({ code: `
from pathlib import Path
import sys
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient

paths = AppPaths.resolve(instance_root=Path(sys.argv[1]))
print(HostClient(paths.host).ping()["pid"])
` }, [home]);
  return processId(stdout, 'host');
}

export function processExists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ESRCH') return false;
    throw error;
  }
}

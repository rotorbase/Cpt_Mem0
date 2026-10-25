# Installation on Windows

The app is a folder with an `.exe` in it. Download the zip from the
releases, unpack it anywhere, and run `Cpt_Mem0.exe`.

**The llama.cpp build is not in the zip.** It is 700 MB with the CUDA
runtime, and `Cpt_Mem0.exe init` downloads it separately, pinned to an
exact build. It belongs on the disk of the machine, not in a file
every user has to download.

## What the package contains

```
Cpt_Mem0/
  Cpt_Mem0.exe          the app
  _internal/            the Python runtime and the libraries
```

**Why a folder and not a single file.** PyInstaller's onefile mode
extracts the whole package to a temporary directory on every start.
That is a slow first start, and a lot of antivirus software reports
such a file. A folder starts in a second and is not reported.

## The first start

```
Cpt_Mem0.exe init
```

That installs the pinned `llama.cpp` build - the binaries plus the
CUDA runtime - into `%LOCALAPPDATA%\cpt-mem0\runtime\`. It does not
change any system setting: no driver, no PATH, no registry. The
system llama.cpp stays untouched.

Then:

```
Cpt_Mem0.exe
```

The window opens. The big switch is on the right.

## The autostart

The switch `AUTOSTART (WINDOWS)` in the footer writes a scheduled
task `Cpt_Mem0` that runs at the logon.

**The task starts the daemon with the models off.** The window and
the API are there, the VRAM stays free, and the models are loaded
when the switch is pressed. Nothing loads a model on its own.

To remove it:

```
schtasks /Delete /TN "Cpt_Mem0" /F
```

## The native window

The window is a **WebView2** - the engine Edge already ships and
every Windows 10 and 11 has. So the app does not bring its own
Chromium: the package stays small.

If the WebView2 runtime is missing, Windows shows the standard
installer for it, or the app falls back to the system browser. The
interface is the same either way.

## Where the data is

```
%LOCALAPPDATA%\cpt-mem0\
  runtime\llama\b11838\    the llama.cpp build
  config\                 the configuration and the window settings
  data\                   the store
  logs\                   the logs of the servers
```

The store has an exclusive lock: one process at a time. A second
instance fails with a message that names the lock.

## The checksum

Every release has a `.sha256` file next to the zip:

```
certutil -hashfile Cpt_Mem0-0.1.0-windows-x64.zip SHA256
```

## Uninstalling

Delete the folder. Delete `%LOCALAPPDATA%\cpt-mem0` to remove the
runtime, the configuration and the store with it.

Nothing was written outside those two places. The scheduled task has
to be deleted by hand - the switch in the footer does it.

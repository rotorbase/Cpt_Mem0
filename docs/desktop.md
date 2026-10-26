# The desktop app

Cpt Mem0 is a Windows program with a window of its own. The window is
made by Electron, the work is done by the Python backend.

## Why Electron

The first attempt used pywebview, which drives the WebView2 of
Windows from Python. That path needs pythonnet, and pythonnet needs a
.NET runtime that carries WinForms - while the WebView2 assemblies
are .NET Framework libraries and do not run on .NET Core. Six
attempts later it was clear that the path is not reliable.

Electron brings its own browser. It does not depend on the .NET
version of the machine, it opens a window on every Windows, and it is
the same technology Discord, VS Code and Slack use.

The interface itself is unchanged: HTML, CSS and JavaScript, served by
the backend. Only the frame around it is new.

## How the pieces fit together

```
    Cpt Mem0.exe (Electron)
        |
        |  starts the backend, waits for it, shows the window
        v
    Cpt_Mem0.exe (Python, in the resources)
        |
        |  owns the store, the models, the API
        v
    http://127.0.0.1:8765
```

The backend is started with `ui --no-browser`: it opens no window and
no browser, it serves and stays alive. The Electron window shows that
address.

**The models are off when the app starts.** The VRAM belongs to
whoever wants it - the big switch in the window turns the models on.
The store is reachable whether the models run or not, so the app is
usable while the GPU is busy with a game.

## The build

Two steps, and the second one needs the first:

```
python packaging/build_windows.py      # the Python backend -> dist/Cpt_Mem0
cd desktop && npm install && npm run dist
```

The result is in `desktop/dist-electron`:

| File | What it is |
| --- | --- |
| `Cpt Mem0 Setup 0.1.0.exe` | the installer, with a start menu entry |
| `Cpt_Mem0-0.1.0-portable.exe` | one file, runs from anywhere |
| `win-unpacked/` | the program itself, without a setup |

## The window has no console

Two things would open a console on the desktop, and both are turned
off:

- The backend is started by Electron with `windowsHide: true` and
  `stdio: ignore`. It is a program, not a command in a terminal.
- The llama.cpp server is a console program itself. It is started
  with `CREATE_NO_WINDOW`, so no black box appears while a model
  loads.

A console that appears and disappears is the one thing that makes an
app feel broken, so it is checked on every start.

## When the window does not open

The log has the answer: `%LOCALAPPDATA%\cpt-mem0\logs\window.log`

The backend writes its own failures there. A daemon that does not
come up throws inside its thread, and without the log the app would
only say that it did not come up.

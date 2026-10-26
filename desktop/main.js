/* The window of Cpt Mem0.
 *
 * The backend is the Python program - it owns the store, the models
 * and the API. This is the frame around it: a window that shows the
 * interface the backend serves.
 *
 * **The backend is started with the models off.** The VRAM belongs
 * to whoever wants it; the big switch in the window turns the models
 * on. The window is there from the first second, and the store is
 * reachable whether the models run or not.
 */

const { app, BrowserWindow, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const PORT = 8765;
const ADDRESS = `http://127.0.0.1:${PORT}`;

let backend = null;
let mainWindow = null;

/** Where the Python backend is.
 *
 * In the packaged app it is in the resources, in the development
 * tree it is the one that was built next to the source.
 */
function backendPath() {
  const packaged = path.join(process.resourcesPath || "", "backend");
  const built = path.join(__dirname, "..", "dist", "Cpt_Mem0");
  const fs = require("fs");
  for (const folder of [packaged, built]) {
    const exe = path.join(folder, "Cpt_Mem0.exe");
    if (fs.existsSync(exe)) return exe;
  }
  return null;
}

/** Wait until the backend answers. */
function waitForBackend(attempts = 120) {
  return new Promise((resolve, reject) => {
    const tryOnce = (left) => {
      const request = http.get(`${ADDRESS}/status`, (response) => {
        response.resume();
        resolve();
      });
      request.on("error", () => {
        if (left <= 0) {
          reject(new Error("the backend did not answer"));
          return;
        }
        setTimeout(() => tryOnce(left - 1), 500);
      });
    };
    tryOnce(attempts);
  });
}

function startBackend() {
  const exe = backendPath();
  if (!exe) {
    console.error("the backend was not found - build it first");
    return false;
  }

  // **No console window.** The backend is a program, not a command
  // in a terminal: without this flag Windows opens a console for it
  // and the user sees a black box blinking on the desktop.
  backend = spawn(exe, ["ui", "--no-browser"], {
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    detached: false,
  });

  backend.stdout.on("data", (data) => {
    process.stdout.write(`[backend] ${data}`);
  });
  backend.stderr.on("data", (data) => {
    process.stderr.write(`[backend] ${data}`);
  });
  backend.on("exit", (code) => {
    console.log(`the backend ended with ${code}`);
  });
  return true;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 760,
    minWidth: 820,
    minHeight: 560,
    backgroundColor: "#0a0b0d",
    title: "Cpt Mem0",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      // **The renderer keeps drawing in the background.** Without it
      // Electron throttles the timers of a window that is covered -
      // the status would stop, and the window would show a stale
      // picture when it comes back.
      backgroundThrottling: false,
    },
  });

  mainWindow.loadURL(ADDRESS);

  // A link goes into the browser, not into the app.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  if (!startBackend()) {
    app.quit();
    return;
  }

  try {
    await waitForBackend();
  } catch (error) {
    console.error(error.message);
  }

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (backend && !backend.killed) backend.kill();
  app.quit();
});

app.on("before-quit", () => {
  if (backend && !backend.killed) backend.kill();
});

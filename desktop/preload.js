/* The bridge between the window and the system.
 *
 * The window may not touch the disk. It asks for a folder, and this
 * file asks the main process - which owns the dialogs.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("cpt", {
  /** Let the user pick the folder with the models. */
  pickModelFolder: () => ipcRenderer.invoke("pick-model-folder"),

  /** Open a page in the browser - a download link, for example. */
  openExternal: (url) => ipcRenderer.invoke("open-external", url),
});

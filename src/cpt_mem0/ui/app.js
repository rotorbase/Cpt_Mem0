/* Cpt Mem0 - the client.
 *
 * The window is a client of the daemon. It never opens the store
 * itself - the store has an exclusive lock, and the daemon is the
 * only one that may hold it.
 *
 * Every request goes to the same origin. The daemon serves this file.
 */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  modelsLoaded: false,
  onGpu: false,
  appVram: 0,
  loading: false,
  waiting: "",
  suspended: false,
  activeTopic: null,
  clusters: [],
  pollTimer: null,
};

/* ------------------------------------------------------------------ */
/* das API                                                              */
/* ------------------------------------------------------------------ */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = { ok: false, error: `HTTP ${response.status}` };
  }
  if (!response.ok) {
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  return body;
}

/* ------------------------------------------------------------------ */
/* die Anzeige                                                          */
/* ------------------------------------------------------------------ */

function say(text, kind = "system") {
  const element = document.createElement("div");
  element.className = `msg ${kind}`;
  element.textContent = text;
  $("messages").appendChild(element);
  $("messages").scrollTop = $("messages").scrollHeight;
  return element;
}

function sayWithFacts(text, facts) {
  const element = say(text, "user");
  if (facts && facts.length) {
    const list = document.createElement("ul");
    list.className = "facts";
    for (const fact of facts) {
      const item = document.createElement("li");
      item.textContent = fact;
      list.appendChild(item);
    }
    element.appendChild(list);
  }
  return element;
}

function status(text, kind = "") {
  const element = $("statusline");
  element.textContent = text;
  element.className = `statusline ${kind}`;
}

/* ------------------------------------------------------------------ */
/* die Themen-Tasten                                                     */
/* ------------------------------------------------------------------ */

function renderTopics(clusters, colours) {
  const container = $("topics");
  container.innerHTML = "";

  if (!clusters.length) {
    const empty = document.createElement("button");
    empty.className = "topic empty";
    empty.textContent = "Noch keine Themen";
    empty.disabled = true;
    container.appendChild(empty);
    return;
  }

  clusters.forEach((cluster, index) => {
    const button = document.createElement("button");
    button.className = "topic";
    if (cluster.label === state.activeTopic) {
      button.classList.add("active");
    }
    button.style.setProperty("--topic", colours[index] || "#2a2e36");
    button.textContent = cluster.label;

    const count = document.createElement("span");
    count.className = "topic-count";
    count.textContent = cluster.size;
    button.appendChild(count);

    button.title = cluster.facts.join("\n");
    button.addEventListener("click", () => {
      state.activeTopic = cluster.label;
      status(`Thema: ${cluster.label}`);
      renderTopics(clusters, colours);
    });

    container.appendChild(button);
  });
}

/* ------------------------------------------------------------------ */
/* der grosse Schalter                                                   */
/* ------------------------------------------------------------------ */

function renderPower() {
  const button = $("power");

  // **The switch shows what is running, not what was asked for.**
  // A switch that says "AN" while nothing is loaded is a lie the
  // user notices at once - in the task manager the model is not
  // there. The state comes from the servers.
  const on = state.modelsLoaded;
  const loading = state.loading;
  button.classList.toggle("on", on);
  button.classList.toggle("loading", loading);
  button.querySelector(".power-label").textContent =
    on ? "AN" : loading ? "LÄDT" : "AUS";
  button.title = on
    ? "Die Modelle laufen. Klick: sie beenden und die VRAM freigeben."
    : loading
      ? "Die Modelle werden geladen. Das dauert etwa eine Minute."
      : "Die Modelle sind aus. Klick: sie laden.";
}

async function togglePower() {
  // The action follows the state that is shown, not the one that was
  // asked for. Otherwise a failed load would switch the switch back
  // and the user would not know what is running.
  const action = state.modelsLoaded ? "suspend" : "resume";
  $("power").disabled = true;
  try {
    await api(`/runtime/${action}`, { method: "POST" });
    await refresh();
    if (state.modelsLoaded) {
      const mib = state.appVram ? ` - ${state.appVram} MiB auf der Karte` : "";
      status(
        state.onGpu
          ? `Die Modelle sind geladen${mib}.`
          : "Die Modelle laufen, aber nicht auf der Grafikkarte.",
        state.onGpu ? "ok" : "error");
    } else {
      status("Die Modelle sind aus. Das VRAM ist frei, die Erinnerungen bleiben.",
             "ok");
    }
  } catch (error) {
    await refresh();
    status(error.message, "error");
  } finally {
    $("power").disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* was beim Laden passiert                                               */
/* ------------------------------------------------------------------ */

let lastWaiting = "";

function renderLoading() {
  const text = state.loading
    ? (state.waiting
        ? `Warte auf VRAM: ${state.waiting}`
        : "Die Modelle werden geladen. Das dauert etwa eine Minute.")
    : "";
  if (text === lastWaiting) return;
  lastWaiting = text;
  if (text) {
    status(text, "wait");
  } else if (state.modelsLoaded) {
    const mib = state.appVram ? ` - ${state.appVram} MiB auf der Karte` : "";
    status(state.onGpu
      ? `Die Modelle sind geladen${mib}.`
      : "Die Modelle laufen, aber nicht auf der Grafikkarte.",
      state.onGpu ? "ok" : "error");
  }
}

/* ------------------------------------------------------------------ */
/* die Anzeige aktualisieren                                             */
/* ------------------------------------------------------------------ */

async function refresh() {
  try {
    const runtime = await api("/runtime");
    state.suspended = runtime.suspended;
    state.modelsLoaded = runtime.models_loaded === true;
    state.onGpu = runtime.on_gpu === true;
    state.appVram = runtime.app_vram_mib || 0;
    state.loading = runtime.loading === true;
    state.waiting = runtime.waiting || "";
    renderPower();

    const used = runtime.gpu_used_mib;
    const free = runtime.gpu_free_mib;
    $("vram").textContent = used === null ? "–" : `${used} MiB`;
    // **What this app holds.** The number of the whole card includes
    // every other program, and the user cannot tell from it whether
    // the model is really there.
    $("vram-app").textContent = state.appVram
      ? `${state.appVram} MiB`
      : "–";
    const total = used !== null && free !== null ? used + free : 0;
    const percent = total ? Math.round((used / total) * 100) : 0;
    $("vram-bar").style.width = `${percent}%`;

    const statusData = await api("/status");
    $("memories").textContent = statusData.memories ?? "–";
    $("build").textContent = runtime.servers.length
      ? runtime.servers.map((s) => s.name).join(" + ")
      : "keine Modelle";
    const models = $("models");
    if (models) {
      models.textContent = state.modelsLoaded
        ? (state.onGpu ? "auf der GPU" : "auf der CPU")
        : state.loading ? "lädt" : "aus";
    }
    renderLoading();
  } catch (error) {
    status(error.message, "error");
  }
}

async function refreshTopics() {
  try {
    const count = $("cluster-count").value;
    const data = await api(`/topics?count=${count}`);
    state.clusters = data.clusters || [];
    renderTopics(state.clusters, data.colours || []);
  } catch (error) {
    // The topics need the vectors, not the models. If it fails, the
    // status line says why.
    status(error.message, "error");
  }
}

async function refreshModels() {
  try {
    const data = await api("/plan");
    const select = $("model");
    select.innerHTML = "";
    for (const model of data.models || []) {
      const option = document.createElement("option");
      option.value = model.model_id;
      option.textContent = model.fits
        ? `${model.model_id} ${model.quantisation || ""} · ${model.total_vram_mib} MiB`
        : `${model.model_id} · passt nicht`;
      option.disabled = !model.fits;
      select.appendChild(option);
    }
    const fitting = (data.models || []).filter((m) => m.fits);
    if (data.reason) {
      // No GPU - the window has to say that, not show an empty list.
      $("models").textContent = "keine GPU";
      status(data.reason, "error");
    } else {
      $("models").textContent = fitting.length
        ? `${fitting.length} passend`
        : "kein Modell passt";
    }
  } catch (error) {
    status(error.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* der Chat                                                              */
/* ------------------------------------------------------------------ */

async function send(text) {
  if (!text.trim()) return;
  sayWithFacts(text, null);

  if (state.suspended) {
    say("Die Modelle sind aus. Schalte sie oben ein, dann kann ich antworten.",
        "system");
    return;
  }

  // **The answer takes a moment.** A 9B on a consumer card needs a
  // second or two for the first tokens, and the line shows that
  // something is happening instead of standing empty.
  const waiting = say("…", "assistant waiting");

  try {
    // **The chat, not the write.** The model answers, and the
    // exchange goes into the memory.
    const result = await api("/chat", {
      method: "POST",
      body: JSON.stringify({ text }),
    });

    waiting.remove();
    say(result.answer || "Keine Antwort.", "assistant");

    const facts = result.results || [];
    if (facts.length) {
      const element = say("Gemerkt:", "system");
      const list = document.createElement("ul");
      list.className = "facts";
      for (const item of facts) {
        const line = document.createElement("li");
        line.textContent = item.memory || item;
        list.appendChild(line);
      }
      element.appendChild(list);
    } else {
      say("Dazu gab es nichts zu merken.", "system");
    }
    await refresh();
    await refreshTopics();
  } catch (error) {
    waiting.remove();
    say(error.message, "system");
    status(error.message, "error");
  }
}

async function recall(query) {
  if (!query.trim()) return;
  say(query, "user");
  if (state.suspended) {
    say("Die Modelle sind aus, für die Suche brauche ich sie.", "system");
    return;
  }
  try {
    const result = await api(
      `/search?q=${encodeURIComponent(query)}&limit=8`);
    const results = result.results || [];
    if (!results.length) {
      say("Nichts gefunden.", "system");
      return;
    }
    const element = say("Gefunden:", "system");
    const list = document.createElement("ul");
    list.className = "facts";
    for (const item of results) {
      const line = document.createElement("li");
      line.textContent = item.memory || item;
      list.appendChild(line);
    }
    element.appendChild(list);
  } catch (error) {
    say(error.message, "system");
  }
}

/* ------------------------------------------------------------------ */
/* die Schalter                                                          */
/* ------------------------------------------------------------------ */

function bindToggle(id, onChange) {
  const button = $(id);
  button.addEventListener("click", async () => {
    const next = button.getAttribute("aria-checked") !== "true";
    button.setAttribute("aria-checked", String(next));
    try {
      await onChange(next);
    } catch (error) {
      button.setAttribute("aria-checked", String(!next));
      status(error.message, "error");
    }
  });
}

/* ------------------------------------------------------------------ */
/* der Start                                                             */
/* ------------------------------------------------------------------ */

function start() {
  $("power").addEventListener("click", togglePower);
  renderPower();

  say("Cpt Mem0 ist bereit. Alles, was du mir sagst, wird gemerkt.",
      "system");
  say("Die Themen oben füllen sich aus dem Store. Die grossen Tasten " +
      "schalten die Modelle ein und aus - aus ist die VRAM frei, " +
      "die Erinnerungen bleiben.", "system");

  $("composer").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("input");
    const text = input.value;
    input.value = "";
    await send(text);
  });

  $("cluster-count").addEventListener("input", (event) => {
    $("cluster-value").textContent = event.target.value;
  });
  $("cluster-count").addEventListener("change", refreshTopics);

  bindToggle("autostart", async (next) => {
    // The daemon owns the autostart. It writes the task on Windows.
    await api("/settings/autostart", {
      method: "POST",
      body: JSON.stringify({ enabled: next }),
    });
    status(next ? "Autostart eingeschaltet." : "Autostart aus.");
  });

  bindToggle("voice", async (next) => {
    await api("/settings/voice", {
      method: "POST",
      body: JSON.stringify({ enabled: next }),
    });
    status(next
      ? "Sprachausgabe an. Die Stimme läuft auf der CPU."
      : "Sprachausgabe aus.");
  });

  bindToggle("theme", async (next) => {
    document.body.classList.toggle("plain", !next);
    await api("/settings/theme", {
      method: "POST",
      body: JSON.stringify({ winamp: next }),
    });
  });

  $("wipe").addEventListener("click", async () => {
    if (!confirm("Alle Erinnerungen löschen? Das kann nicht rückgängig gemacht werden.")) {
      return;
    }
    try {
      await api("/memories/reset", { method: "DELETE" });
      say("Der Store ist leer.", "system");
      await refresh();
      await refreshTopics();
    } catch (error) {
      status(error.message, "error");
    }
  });

  $("model").addEventListener("change", async (event) => {
    const model = event.target.value;
    try {
      await api("/settings/model", {
        method: "POST",
        body: JSON.stringify({ model }),
      });
      status(`Für den nächsten Start gewählt: ${model}`);
    } catch (error) {
      status(error.message, "error");
    }
  });

  refresh();
  refreshModels();
  refreshTopics();

  // The VRAM and the store change on their own.
  state.pollTimer = setInterval(refresh, 4000);
}

document.addEventListener("DOMContentLoaded", start);

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
  // The eighteen topics with what is in them.
  board: null,
  modelsLoaded: false,
  onGpu: false,
  appVram: 0,
  loading: false,
  waiting: "",
  // **The thread.** The model needs to see the last turns, otherwise
  // every question stands alone and it is not a conversation.
  history: [],
  // **The topics the user marked.** They stay at the front.
  favourites: [],
  theme: "plain",
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

  // **The board of the topics.** The names are fixed and the store
  // fills them: a topic without anything in it is white, a topic
  // with something in it lights up in its own colour.
  const board = state.board;
  if (board && board.topics) {
    renderBoard(board);
    return;
  }


  // **Twenty fields, all of them visible.** The user wants to see
  // the whole conversation of the store at a glance - the fields
  // that are not filled yet are placeholders, so the row does not
  // jump when a new topic appears, and the empty store looks the
  // same as a full one. The topics come first - the placeholders
  // fill the rest.
  const wanted = Math.max(18, clusters.length);
  const placeholders = document.createDocumentFragment();
  for (let empty = clusters.length; empty < wanted; empty++) {
    const slot = document.createElement("div");
    slot.className = "topic empty-slot";
    slot.title = "Noch kein Thema";
    placeholders.appendChild(slot);
  }

  if (!clusters.length) {
    const note = document.createElement("div");
    note.className = "topics-note";
    note.textContent = "Noch keine Themen - der Store ist leer.";
    container.appendChild(note);
  }

  // **The six that are being played shine, the rest glow quiet.**
  // The topics come in the order of their strength, so the first
  // six are the ones that matter right now.
  clusters.forEach((cluster, index) => {
    const button = document.createElement("button");
    button.className = "topic";
    if (cluster.label === state.activeTopic) {
      button.classList.add("active");
    }
    button.style.setProperty("--topic",
      cluster.colour || colours[index] || "#2a2e36");
    button.style.setProperty("--topic-fill",
      cluster.fill || "transparent");
    // **The six that are being played shine full, the rest glow
    // quiet.** Like the keys of a synthesizer: the ones that are
    // active are lit, the others are there but dark.
    const lit = index < 6;
    button.style.setProperty("--topic-glow",
      lit ? "0 0 14px var(--topic), 0 0 34px var(--topic)"
          : "0 0 6px var(--topic)");
    button.classList.toggle("lit", lit);

    const label = document.createElement("span");
    label.className = "topic-label";
    label.textContent = cluster.label;
    button.appendChild(label);

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

  // **The placeholders fill the rest of the row.** The topics come
  // first - the ones that are being played are the ones at the top
  // left, like the keys under the fingers.
  container.appendChild(placeholders);
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
    // **The topics are asked for again.** The names come from the
    // model - and the model only runs when the switch is on. Without
    // this the buttons keep the names of a store without a model:
    // the first words of the fact, in the language of the fact.
    await refreshTopics();
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
/* die Lieblingsthemen                                                   */
/* ------------------------------------------------------------------ */

function renderFavourites() {
  // **The stars are gone.** The topics are all shown at once, so
  // there is nothing to mark and nothing to keep in front.
}

/* ------------------------------------------------------------------ */
/* die Oberflaeche                                                       */
/* ------------------------------------------------------------------ */

function renderTheme() {
  // **One interface.** The others were tried and the plain one was
  // the one that stayed.
  document.body.className = "plain";
}

/* ------------------------------------------------------------------ */
/* das Cockpit                                                           */
/* ------------------------------------------------------------------ */

function renderCockpit(runtime) {
  // **The instruments.** The speed of the answer, how busy the card
  // is, how much of it is in use, and whether the model runs.
  const tps = runtime.tokens_per_second;
  $("tps").textContent = tps ? Number(tps).toFixed(1) : "–";

  const percent = runtime.gpu_percent;
  $("gpu-percent").textContent =
    percent === null || percent === undefined ? "–" : `${percent}%`;
  // **The short name of the card.** "NVIDIA GEFORCE RTX 3060" does
  // not fit into a dial, and the panel has to stay inside the
  // window.
  const card = String(runtime.gpu_name || "GPU");
  $("gpu-name").textContent = card
    .replace(/NVIDIA\s*/gi, "")
    .replace(/GeForce\s*/gi, "")
    .trim() || "GPU";

  const used = runtime.gpu_used_mib;
  const free = runtime.gpu_free_mib;
  const total = (used || 0) + (free || 0);
  const share = total ? Math.round((used || 0) / total * 100) : 0;
  $("vram-percent").textContent = total ? `${share}%` : "–";
  $("cockpit-vram-bar").style.width = `${share}%`;

  const lamp = $("cockpit-power");
  lamp.classList.toggle("on", runtime.models_loaded === true);
  lamp.classList.toggle("loading", runtime.loading === true);
  $("cockpit-power-label").textContent =
    runtime.loading ? "lädt" : runtime.models_loaded ? "an" : "aus";
}

/* ------------------------------------------------------------------ */
/* die Stimmen                                                           */
/* ------------------------------------------------------------------ */

async function loadVoices() {
  const select = $("voice-name");
  if (!select) return;
  let data;
  try {
    data = await api("/settings/voices");
  } catch (error) {
    status(error.message, "error");
    return;
  }

  select.innerHTML = "";
  if (!data.available) {
    // **The reason belongs to the user.** A switch that changes
    // nothing is worse than no switch.
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "keine Stimme";
    select.appendChild(option);
    select.disabled = true;
    status(data.reason || "Auf diesem Rechner gibt es keine Stimme.",
           "error");
    return;
  }

  select.disabled = false;
  for (const voice of data.voices) {
    const option = document.createElement("option");
    option.value = voice.description;
    option.textContent = `${voice.name} (${voice.language})`;
    select.appendChild(option);
  }
  if (data.selected) {
    select.value = data.selected;
  }
  select.addEventListener("change", async () => {
    try {
      await api("/settings/voice-name", {
        method: "POST",
        body: JSON.stringify({ value: select.value }),
      });
      status(`Stimme: ${select.value}`);
    } catch (error) {
      status(error.message, "error");
    }
  });
}

/* ------------------------------------------------------------------ */
/* das Brett der Themen                                                  */
/* ------------------------------------------------------------------ */

/* **Eighteen topics, three groups.** The names do not change - the
 * user has to recognise the button. A topic with something in it
 * lights up, one without stays white. */
function renderBoard(board) {
  const container = $("topics");
  container.innerHTML = "";
  container.classList.add("board");

  const byGroup = new Map();
  for (const topic of board.topics || []) {
    if (!byGroup.has(topic.group)) {
      byGroup.set(topic.group, []);
    }
    byGroup.get(topic.group).push(topic);
  }

  for (const [group, topics] of byGroup) {
    const section = document.createElement("div");
    section.className = "topic-group";

    const title = document.createElement("div");
    title.className = "topic-group-label";
    title.textContent = group;
    section.appendChild(title);

    const row = document.createElement("div");
    row.className = "topic-row";
    for (const topic of topics) {
      const button = document.createElement("button");
      button.type = "button";
      // **White while it is empty.** A topic lights up as soon as
      // the store has something in it.
      button.classList.add("topic");
      if (!topic.count) {
        button.classList.add("empty-topic");
      } else {
        button.classList.add("filled-topic");
        button.style.setProperty("--topic", topic.colour || "#2a2e36");
        button.style.setProperty("--topic-fill", topic.fill || "transparent");
      }
      button.style.setProperty("--topic-glow", topic.count
        ? "0 0 14px var(--topic), 0 0 34px var(--topic)"
        : "none");

      const label = document.createElement("span");
      label.className = "topic-label";
      label.textContent = topic.name;
      button.appendChild(label);

      const count = document.createElement("span");
      count.className = "topic-count";
      count.textContent = topic.count;
      button.appendChild(count);

      button.title = topic.count
        ? `${topic.count} Erinnerungen in ${topic.name}`
        : `Noch nichts in ${topic.name}`;
      button.addEventListener("click", () => openTopic(topic));
      row.appendChild(button);
    }
    section.appendChild(row);
    container.appendChild(section);
  }
}

/* **What is in this topic.** The window shows the groups of facts -
 * the sub-themes - and every single one of them. */
function openTopic(topic) {
  const dialog = $("topic-dialog");
  const title = $("topic-dialog-title");
  const hint = $("topic-dialog-hint");
  const body = $("topic-dialog-body");
  if (!dialog) return;

  title.textContent = topic.name;
  hint.textContent = topic.count
    ? `${topic.count} Erinnerungen im Store`
    : "In diesem Thema ist noch nichts gespeichert.";

  body.innerHTML = "";
  if (!topic.count) {
    const empty = document.createElement("p");
    empty.className = "model-hint";
    empty.textContent =
      "Sobald du etwas erzählst, was hierher passt, "
      + "erscheint es in diesem Fenster.";
    body.appendChild(empty);
  } else {
    for (const part of topic.subtopics || []) {
      const block = document.createElement("div");
      block.className = "subtopic";

      const head = document.createElement("div");
      head.className = "subtopic-head";
      const name = document.createElement("span");
      name.className = "subtopic-name";
      name.textContent = part.label;
      const size = document.createElement("span");
      size.className = "subtopic-size";
      size.textContent = `${part.size}`;
      head.appendChild(name);
      head.appendChild(size);
      block.appendChild(head);

      const list = document.createElement("ul");
      list.className = "subtopic-facts";
      for (const fact of part.facts || []) {
        const item = document.createElement("li");
        item.textContent = fact;
        list.appendChild(item);
      }
      block.appendChild(list);
      body.appendChild(block);
    }
  }
  dialog.showModal();
}

/* ------------------------------------------------------------------ */
/* der Themenbaum                                                        */
/* ------------------------------------------------------------------ */

/* **Everything in one place.** The buttons show the topics, small
 * and short. The tree shows the whole network: every topic with the
 * facts under it, and how many there are. */
async function openTree() {
  const dialog = $("tree-dialog");
  const tree = $("tree");
  const hint = $("tree-hint");
  if (!dialog) return;

  try {
    const data = await api("/topic-board");
    const clusters = [];
    for (const topic of data.topics || []) {
      for (const part of topic.subtopics || []) {
        clusters.push({
          label: `${topic.name} · ${part.label}`,
          size: part.size,
          facts: part.facts,
        });
      }
    }
    hint.textContent = `${(data.topics || []).length} Themen · ${data.facts || 0} Fakten`;
    tree.innerHTML = "";
    for (const cluster of clusters) {
      const branch = document.createElement("div");
      branch.className = "branch";

      const head = document.createElement("div");
      head.className = "branch-head";
      const name = document.createElement("span");
      name.className = "branch-name";
      name.textContent = cluster.label;
      const count = document.createElement("span");
      count.className = "branch-count";
      count.textContent = cluster.size;
      head.appendChild(name);
      head.appendChild(count);
      branch.appendChild(head);

      const list = document.createElement("ul");
      list.className = "branch-facts";
      for (const fact of cluster.facts || []) {
        const item = document.createElement("li");
        item.textContent = fact;
        list.appendChild(item);
      }
      branch.appendChild(list);
      tree.appendChild(branch);
    }
    if (!clusters.length) {
      const empty = document.createElement("p");
      empty.className = "model-hint";
      empty.textContent = "Im Store ist noch nichts.";
      tree.appendChild(empty);
    }
  } catch (error) {
    hint.textContent = error.message;
  }
  dialog.showModal();
}

/* ------------------------------------------------------------------ */
/* die Modellvorschlaege                                                 */
/* ------------------------------------------------------------------ */

async function showModels() {
  const dialog = $("model-dialog");
  const list = $("model-list");
  const hint = $("model-hint");
  if (!dialog) return;

  try {
    const data = await api("/models");
    if (data.error) {
      hint.textContent = data.error;
    } else {
      hint.textContent =
        `${data.card.name || "Die Karte"} · ${data.card.vram_mib} MiB. `
        + "Der Download führt zu HuggingFace - die Datei dann in den "
        + "Modellordner legen.";
    }
    list.innerHTML = "";
    for (const model of data.models || []) {
      const item = document.createElement("div");
      item.className = model.local ? "model-item local" : "model-item";

      const name = document.createElement("div");
      const title = document.createElement("div");
      title.className = "model-name";
      title.textContent = `${model.model_id} ${model.quantisation}`;
      if (model.local) {
        const badge = document.createElement("span");
        badge.className = "model-badge";
        badge.textContent = "auf der Platte";
        title.appendChild(badge);
      } else if (!model.measured) {
        const badge = document.createElement("span");
        badge.className = "model-badge unmeasured";
        badge.textContent = "ungeprüft";
        title.appendChild(badge);
      }
      const meta = document.createElement("div");
      meta.className = "model-meta";
      meta.textContent = `${model.size_mib} MiB Datei`;
      name.appendChild(title);
      name.appendChild(meta);

      const numbers = document.createElement("div");
      numbers.className = "model-numbers";
      numbers.textContent = `${model.vram_mib} MiB VRAM · n_ctx ${model.n_ctx}`;

      const link = document.createElement("a");
      link.href = model.link;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = model.local ? "HuggingFace" : "Herunterladen";

      item.appendChild(name);
      item.appendChild(numbers);
      item.appendChild(link);
      list.appendChild(item);
    }
  } catch (error) {
    hint.textContent = error.message;
  }
  dialog.showModal();
}

/* ------------------------------------------------------------------ */
/* die Anzeige aktualisieren                                             */
/* ------------------------------------------------------------------ */

async function refresh() {
  try {
    const runtime = await api("/runtime");
    // **The names come with the models.** While the models are off,
    // a topic is called after the first words of its fact - in the
    // language of the fact. When the models come up, the names are
    // asked for again, and they are German.
    const wereLoaded = state.modelsLoaded;
    state.suspended = runtime.suspended;
    state.modelsLoaded = runtime.models_loaded === true;
    if (state.modelsLoaded && !wereLoaded) {
      await refreshTopics();
    }
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
    renderCockpit(runtime);
  } catch (error) {
    status(error.message, "error");
  }
}

async function refreshTopics() {
  try {
    // **The board.** The eighteen topics with what is in them - and
    // the names of the groups of facts, for the tree.
    const data = await api("/topic-board");
    state.board = data;
    renderBoard(data);
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
    // **The model that is chosen is the one that is shown.** The
    // list is sorted by what fits, and the chosen one can be
    // anywhere in it.
    try {
      const settings = await api("/settings");
      const chosen = settings.model;
      if (chosen) {
        for (const option of select.options) {
          if (option.value === chosen) {
            select.value = chosen;
            break;
          }
        }
      }
    } catch (error) {
      // The list is shown anyway.
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
      body: JSON.stringify({ text, history: state.history }),
    });

    waiting.remove();
    const answer = result.answer || "Keine Antwort.";
    say(answer, "assistant");

    // **The thread grows with the conversation.** Only the last turns
    // are kept - the model gets a window, not the whole life.
    state.history.push({ role: "user", content: text });
    state.history.push({ role: "assistant", content: answer });
    if (state.history.length > 12) {
      state.history = state.history.slice(-12);
    }

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

async function start() {
  $("power").addEventListener("click", togglePower);
  renderPower();

  // **The settings come first.** The interface and the topics the
  // user marked are part of them, and both are shown in the first
  // frame.
  try {
    const settings = await api("/settings");
    state.favourites = settings.favourites || [];
    state.theme = settings.theme || "plain";
    renderTheme();
    renderFavourites();
  } catch (error) {
    status(error.message, "error");
  }

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

  bindToggle("voice", async (next) => {
    await api("/settings/voice", {
      method: "POST",
      body: JSON.stringify({ value: next }),
    });
    status(next
      ? "Sprachausgabe an. Die Stimme läuft auf der CPU."
      : "Sprachausgabe aus.");
  });

  loadVoices();

  $("show-models").addEventListener("click", showModels);
  $("open-tree").addEventListener("click", openTree);
  $("tree-close").addEventListener("click", () => {
    $("tree-dialog").close();
  });
  $("model-close").addEventListener("click", () => {
    $("model-dialog").close();
  });
  $("topic-dialog-close").addEventListener("click", () => {
    $("topic-dialog").close();
  });

  // **The folder with the models.** The user puts the GGUF files
  // where they want them, and the app looks there.
  $("pick-folder").addEventListener("click", async () => {
    try {
      const folder = await window.cpt.pickModelFolder();
      if (!folder) return;
      await api("/settings/model-folder", {
        method: "POST",
        body: JSON.stringify({ value: folder }),
      });
      $("model-folder-path").textContent = folder;
      $("model-folder-path").title = folder;
      status(`Modelle werden hier gesucht: ${folder}`);
    } catch (error) {
      status(error.message, "error");
    }
  });

  // **The switches show what is set.** Before this they were
  // whatever the HTML said, and the state came back from the server
  // on every start.
  api("/settings").then((settings) => {
    for (const name of ["autostart", "voice"]) {
      const button = $(name);
      if (button) {
        button.setAttribute("aria-checked",
                            String(settings[name] === true));
      }
    }
  }).catch(() => {});

  // **The store is not wiped by one click.**
  //
  // The native question box of the browser is not usable here: in
  // the packaged app it is a dialog that does not belong to the
  // window, and it can freeze the interface. So the button asks
  // twice - the first click arms it, the second one wipes. Six
  // seconds without a second click and it disarms itself.
  let wipeArmed = false;
  let wipeTimer = null;
  $("wipe").addEventListener("click", async () => {
    const button = $("wipe");
    if (!wipeArmed) {
      wipeArmed = true;
      button.textContent = "Wirklich alle löschen?";
      button.classList.add("armed");
      status("Noch ein Klick löscht alle Erinnerungen. Das geht nicht "
             + "rückgängig.", "error");
      if (wipeTimer) clearTimeout(wipeTimer);
      wipeTimer = setTimeout(() => {
        wipeArmed = false;
        button.textContent = "Alle Erinnerungen löschen";
        button.classList.remove("armed");
      }, 6000);
      return;
    }

    wipeArmed = false;
    if (wipeTimer) clearTimeout(wipeTimer);
    button.textContent = "Alle Erinnerungen löschen";
    button.classList.remove("armed");
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

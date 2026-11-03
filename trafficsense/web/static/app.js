// TrafficSense dashboard client: live status over WebSocket + override commands.
// Commands issued while the link is down are queued and re-sent on reconnect
// (communication failsafe, Table 2 of the paper).
(() => {
  const queue = [];
  let ws, nextId = 1;

  function connect() {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onopen = () => {
      setConn(true);
      while (queue.length) ws.send(JSON.stringify(queue.shift()));
    };
    ws.onclose = (e) => {
      setConn(false);
      if (e.code === 4401) { location.href = "/login"; return; }
      setTimeout(connect, 2000);
    };
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "status") render(msg.data);
      else if (msg.type === "ack") note(`Command "${msg.action}" applied.`);
      else if (msg.type === "error") note(`Error: ${msg.message}`, true);
    };
  }

  function send(cmd) {
    cmd.id = nextId++;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd));
    else { queue.push(cmd); note(`Offline — "${cmd.action}" queued, will be sent on reconnect.`); }
  }

  const $ = (id) => document.getElementById(id);
  const setText = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  const setConn = (ok) => { const el = $("conn"); if (el) el.className = "dot " + (ok ? "ok" : "bad"); };
  function note(text, bad) {
    const el = $("cmd-msg");
    if (el) { el.textContent = text; el.className = bad ? "error" : "muted"; }
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const light = (color) =>
    `<div class="light ${color}"><i class="r"></i><i class="y"></i><i class="g"></i></div>`;
  const classes = (obj) => Object.entries(obj || {}).map(([k, v]) => `${esc(k)}: ${v}`).join(", ") || "none";

  function render(s) {
    const c = s.controller, st = s.stats;
    setText("s-mode", c.mode);
    setText("s-phase", `${c.lane} · ${c.phase.replace("_", " ")} · ${c.elapsed}s`);
    setText("s-stops", `${st.stops} (+${st.skips} skips)`);
    setText("s-saved", `${st.saved_green} s`);
    setText("s-wait", `${st.avg_wait} s`);
    setText("s-faults", st.faults);

    const lanes = $("lanes");
    if (lanes) {
      lanes.innerHTML = s.lanes.map((l) => {
        let body;
        if (!l.monitored) body = `<p class="muted">No camera · fixed timing</p>`;
        else body = `
          <p><b>${l.occupied ? "Vehicle present" : "Empty"}</b> · ${l.count} in zone</p>
          <p class="muted">Now: ${classes(l.class_counts)}</p>
          <p class="muted">Counted: ${l.line_in + l.line_out} crossed · seen ${classes(l.seen)}</p>
          <p class="muted">${l.fps} FPS · ${l.infer_ms} ms/frame · conf ${l.avg_conf}</p>
          ${l.healthy ? "" : `<p class="error">FAILSAFE: ${esc(l.fault)}</p>`}`;
        const active = l.id === c.lane && c.phase === "green" ? ` · ${c.elapsed}/${c.planned_green}s` +
          (c.empty_for ? ` · empty ${c.empty_for}s` : "") : "";
        return `<div class="card lane ${l.light}">${light(l.light)}<div>
          <h3>${esc(l.id)} · ${esc(l.name)}${active}</h3>${body}</div></div>`;
      }).join("");
    }
    for (const l of s.lanes) {
      const kv = $(`cam-${l.id}`);
      if (kv) kv.innerHTML = `
        <dt>Signal</dt><dd>${l.light}</dd><dt>Lane status</dt><dd>${l.occupied ? "vehicle present" : "empty"}</dd>
        <dt>In zone</dt><dd>${l.count} (${classes(l.class_counts)})</dd>
        <dt>Line crossings</dt><dd>in ${l.line_in} · out ${l.line_out}</dd>
        <dt>Unique vehicles</dt><dd>${classes(l.seen)}</dd>
        <dt>Avg. confidence</dt><dd>${l.avg_conf}</dd><dt>Speed</dt><dd>${l.fps} FPS · ${l.infer_ms} ms</dd>
        <dt>Health</dt><dd>${l.healthy ? "OK" : "FAILSAFE — " + esc(l.fault)}</dd>`;
      const ctl = $(`ctl-light-${l.id}`);
      if (ctl) ctl.className = `light ${l.light}`;
    }
  }

  document.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-cmd]");
    if (!b) return;
    const { cmd, lane, color } = b.dataset;
    if (cmd === "emergency_stop" && !confirm("Set every lane to red?")) return;
    send({ action: cmd, lane, color });
  });

  connect();
})();

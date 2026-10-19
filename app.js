(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const escape = (value) =>
    String(value).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const names = {
    vanilla: "Vanilla",
    cheatsheet: "Dynamic Cheatsheet",
    reasoningbank: "ReasoningBank",
    sleeptime: "Sleep-time Compute",
    rho: "RHO",
    balance: "Balance",
    iteration: "RHO+Iteration",
    lite: "HarnessPro-Lite",
    medium: "HarnessPro-Medium",
    heavy: "HarnessPro-Heavy",
  };
  const modeInfo = {
    lite: {
      name: "Lite",
      priority: "Cost-focused",
      color: "#347da3",
      background: "#eef5f8",
    },
    medium: {
      name: "Medium",
      priority: "Balanced",
      color: "#8262af",
      background: "#f4f0f8",
    },
    heavy: {
      name: "Heavy",
      priority: "Performance-focused",
      color: "#c16b3c",
      background: "#fcf3ec",
    },
  };
  const benchmarks = {
    tb2: "Terminal-Bench 2",
    swe: "SWE-Bench Pro",
    program: "ProgramBench",
  };
  const models = { gpt: "GPT-5.5 × Codex", glm: "GLM-5.2 × Claude Code" };
  const heroState = { model: "gpt", benchmark: "tb2" };
  const money = (n) => "$" + n.toFixed(2);
  const text = (x, y, content, attrs = "") =>
    `<text x="${x}" y="${y}" ${attrs}>${escape(content)}</text>`;
  const line = (x1, y1, x2, y2, attrs = "") =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
  const numberLabel = (n) =>
    n.toFixed((n < 0.1 && n !== 0) || Math.round(n * 10) !== n * 10 ? 2 : 1);

  function chartFrame(svg) {
    const width = Math.round(svg.getBoundingClientRect().width);
    const compact = width < 440;
    const height = compact ? 420 : Math.round(width * 0.58) + 90;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    return {
      width,
      height,
      compact,
      left: 48,
      right: width - 22,
      top: 60,
      bottom: height - 74,
    };
  }

  function positionChartLabels(svg) {
    const { width, height } = svg.viewBox.baseVal;
    const markers = [
      ...svg.querySelectorAll(".chart-marker, .hero-marker"),
    ].map((marker) => marker.getBBox());
    const frontier = svg.querySelector(".chart-frontier");
    const pathPoints = frontier
      ? Array.from(
          { length: Math.ceil(frontier.getTotalLength()) + 1 },
          (_, i) => frontier.getPointAtLength(i),
        )
      : [];
    const labels = [...svg.querySelectorAll(".plot-label")].sort(
      (a, b) => Number(a.dataset.priority) - Number(b.dataset.priority),
    );
    const placed = [];
    const overlaps = (a, b, gap = 3) =>
      a.x < b.x + b.width + gap &&
      a.x + a.width + gap > b.x &&
      a.y < b.y + b.height + gap &&
      a.y + a.height + gap > b.y;
    for (const label of labels) {
      const cx = Number(label.dataset.x),
        cy = Number(label.dataset.y);
      const box = label.getBBox();
      const candidates =
        label.dataset.method === "heavy"
          ? [
              [cx - box.width / 2, cy - box.height - 18],
              [cx - box.width / 2, cy + 20],
            ]
          : [
              [box.x, box.y],
              [cx - box.width / 2, cy + 17],
              [cx + 13, cy + 13],
              [cx + 13, cy - box.height / 2],
              [cx - box.width - 13, cy - box.height / 2],
              [cx - box.width / 2, cy - box.height - 13],
              [cx - box.width - 12, cy - box.height - 12],
              [cx + 12, cy - box.height - 12],
              [cx - box.width - 12, cy + 16],
              [cx - box.width / 2, cy + 36],
            ];
      let best;
      for (const [rawX, rawY] of candidates) {
        const target = {
          x: Math.max(4, Math.min(width - box.width - 4, rawX)),
          y: Math.max(29, Math.min(height - 56 - box.height, rawY)),
          width: box.width,
          height: box.height,
        };
        const collisions =
          markers.filter((m) => overlaps(target, m)).length +
          placed.filter((b) => overlaps(target, b)).length;
        const lineHit = pathPoints.some(
          (p) =>
            p.x > target.x - 2 &&
            p.x < target.x + target.width + 2 &&
            p.y > target.y - 2 &&
            p.y < target.y + target.height + 2,
        );
        const penalty = collisions * 100 + Number(lineHit) * 10;
        if (!best || penalty < best.penalty) best = { target, penalty };
        if (!penalty) break;
      }
      for (const text of label.querySelectorAll("text")) {
        text.setAttribute(
          "x",
          Number(text.getAttribute("x")) + best.target.x - box.x,
        );
        text.setAttribute(
          "y",
          Number(text.getAttribute("y")) + best.target.y - box.y,
        );
      }
      placed.push(best.target);
    }
  }

  function heroChart() {
    const svg = $("#hero-chart");
    const rows = window.HARNESS_RESULTS[heroState.model][
      heroState.benchmark
    ].filter((r) =>
      ["vanilla", "rho", "lite", "medium", "heavy"].includes(r[0]),
    );
    const benchmark = benchmarks[heroState.benchmark];
    const model = models[heroState.model];
    const heavy = rows.find((r) => r[0] === "heavy");
    const rho = rows.find((r) => r[0] === "rho");
    const vanilla = rows.find((r) => r[0] === "vanilla");
    const lite = rows.find((r) => r[0] === "lite");
    const { left, right, top, bottom, compact } = chartFrame(svg);
    const costs = rows.map((r) => r[2]),
      passes = rows.map((r) => r[1]);
    const range = Math.max(...costs) - Math.min(...costs);
    const xmin = Math.max(0, Math.min(...costs) - range * 0.18);
    const xmax = Math.max(...costs) + range * 0.18;
    const ymin = Math.floor((Math.min(...passes) - 3) / 5) * 5;
    const ymax = Math.ceil((Math.max(...passes) + 3) / 5) * 5;
    const x = (n) => left + ((n - xmin) / (xmax - xmin)) * (right - left);
    const y = (n) => bottom - ((n - ymin) / (ymax - ymin)) * (bottom - top);
    const metric =
      heroState.benchmark === "program"
        ? compact
          ? "Macro-avg. pass rate (%) ↑"
          : "Macro-average pass rate (%) ↑"
        : "Pass rate (%) ↑";
    let s = `<title id="hero-chart-title">${escape(benchmark)} — ${escape(model)}</title><desc id="hero-chart-description">Reported means. ${heroState.benchmark === "program" ? "Pass rate is the macro-average fraction of tests passed per task. " : ""}${rows.map(([id, p, c]) => `${names[id]}: ${p.toFixed(1)} percent at ${money(c)} per task.`).join(" ")}</desc>`;
    const tickStyle =
      'font-family="DM Mono,monospace" font-size="14" fill="#5c5f6c" stroke="none"';
    for (let n = ymin; n <= ymax; n += 5) {
      s += line(
        left,
        y(n),
        right,
        y(n),
        'stroke="#cbcdd8" stroke-width=".8" stroke-dasharray="2 5"',
      );
      s += text(left - 13, y(n) + 3, n, `${tickStyle} text-anchor="end"`);
    }
    const rough = (xmax - xmin) / (compact ? 2 : 4);
    const magnitude = 10 ** Math.floor(Math.log10(rough));
    const step = [1, 2, 2.5, 5, 10]
      .map((n) => n * magnitude)
      .find((n) => n >= rough);
    const decimals = (step.toFixed(6).replace(/0+$/, "").split(".")[1] || "")
      .length;
    for (let n = Math.ceil(xmin / step) * step; n <= xmax + 0.0001; n += step) {
      s += line(
        x(n),
        top,
        x(n),
        bottom,
        'stroke="#cbcdd8" stroke-width=".7" stroke-dasharray="2 5"',
      );
      s += text(
        x(n),
        bottom + 22,
        "$" + n.toFixed(decimals),
        `${tickStyle} text-anchor="middle" class="hero-cost-tick" data-value="${n}"`,
      );
    }
    s += text(
      left,
      top - 19,
      metric,
      'font-family="Manrope,sans-serif" font-size="14" font-weight="600" fill="#5d616e" stroke="none"',
    );
    s += text(
      right,
      bottom + 47,
      "Mean cost per task ($) →",
      'font-family="Manrope,sans-serif" font-size="14" fill="#5d616e" stroke="none" text-anchor="end"',
    );
    const ours = rows.filter((r) => modeInfo[r[0]]).sort((a, b) => a[2] - b[2]);
    s += `<path class="chart-frontier" d="${ours.map((r, i) => `${i ? "L" : "M"}${x(r[2])} ${y(r[1])}`).join(" ")}" fill="none" stroke="#8296df" stroke-width="2" stroke-dasharray="4 5"/>`;
    const closeBaselines =
      Math.hypot(x(vanilla[2]) - x(lite[2]), y(vanilla[1]) - y(lite[1])) < 48;
    for (const [id, p, c, t] of rows) {
      const cx = x(c),
        cy = y(p),
        mode = modeInfo[id];
      const color = mode ? mode.color : id === "rho" ? "#404451" : "#767b8a";
      s += `<g data-hero-method="${id}" data-pass="${p}" data-cost="${c}"><title>${escape(names[id])}: ${p.toFixed(1)}% · ${money(c)}/task · ${t}s</title>`;
      if (id === "rho") {
        s += `<path class="hero-marker" d="M${cx} ${cy - 5}l5 5-5 5-5-5z" fill="${color}" stroke="#f0f3fc" stroke-width="2"/>`;
      } else {
        if (id === "heavy")
          s += `<circle cx="${cx}" cy="${cy}" r="15" fill="#e4cbb5" stroke="none" opacity=".55"/>`;
        s += `<circle class="hero-marker" cx="${cx}" cy="${cy}" r="${id === "heavy" ? 7 : 5}" fill="${color}" stroke="#f0f3fc" stroke-width="2"/>`;
      }
      let tx = cx + 12,
        ty = cy + 4,
        anchor = "start";
      if (id === "lite" || id === "medium") {
        tx = cx;
        ty = cy + (id === "medium" ? 28 : 20);
        anchor = "middle";
      }
      if (id === "heavy") {
        tx = cx;
        ty = cy - 39;
        anchor = "middle";
      }
      if (id === "rho") {
        tx = cx;
        ty = cy + 23;
        anchor = "middle";
      }
      if (closeBaselines && id === "lite") {
        tx = cx - 10;
        ty = cy - 10;
        anchor = "end";
      }
      if (closeBaselines && id === "vanilla") {
        tx = cx + 10;
        ty = cy + 18;
      }
      s += `<g class="plot-label" data-method="${id}" data-x="${cx}" data-y="${cy}" data-priority="${["heavy", "medium", "lite", "rho", "vanilla"].indexOf(id)}">`;
      s += text(
        tx,
        ty,
        mode ? mode.name : names[id],
        `font-family="Manrope,sans-serif" font-size="${id === "heavy" ? 16 : 14}" font-weight="${mode ? 700 : 500}" fill="${color}" stroke="none" text-anchor="${anchor}"`,
      );
      if (id === "heavy")
        s += text(
          cx,
          cy - 15,
          `${p.toFixed(1)}% · ${money(c)}`,
          'class="hero-heavy-value" font-family="DM Mono,monospace" font-size="14" fill="#935330" stroke="none" text-anchor="middle"',
        );
      s += "</g></g>";
    }
    svg.innerHTML = s;
    positionChartLabels(svg);
    svg.setAttribute(
      "aria-labelledby",
      "hero-chart-title hero-chart-description",
    );
    $("#hero-plot-caption").textContent = benchmark;
    $("#hero-plot-model").textContent = model;
    $("#hero-pass-gain").textContent = "+" + (heavy[1] - rho[1]).toFixed(1);
    $("#hero-cost-saving").textContent = (
      (1 - heavy[2] / rho[2]) *
      100
    ).toFixed(1);
  }

  $("#hero-benchmark").addEventListener("change", (event) => {
    heroState.benchmark = event.target.value;
    heroChart();
  });
  $("#hero-model").addEventListener("change", (event) => {
    heroState.model = event.target.value;
    heroChart();
  });

  function renderMainTable() {
    let mobileHtml = "";
    for (const model of Object.keys(models)) {
      const data = window.HARNESS_RESULTS[model];
      const methods = data.tb2.map((r) => r[0]);
      const best = Object.fromEntries(
        Object.keys(benchmarks).map((bench) => [
          bench,
          [
            Math.max(...data[bench].map((r) => r[1])),
            Math.min(...data[bench].map((r) => r[2])),
            Math.min(...data[bench].map((r) => r[3])),
          ],
        ]),
      );
      const methodName = (id) =>
        `${modeInfo[id] ? '<i class="table-mode-dot" aria-hidden="true"></i>' : ""}${escape(names[id])}`;
      const metricCell = (row, bench, metric) => {
        const value = row[metric + 1],
          sd = row[metric + 4];
        const formatted =
          metric === 0
            ? value.toFixed(1)
            : metric === 1
              ? value.toFixed(2)
              : String(value);
        const tag = value === best[bench][metric] ? "strong" : "span";
        const std =
          sd === undefined
            ? ""
            : ` <span class="main-std">± ${metric === 0 ? numberLabel(sd) : metric === 1 ? sd.toFixed(2) : sd}</span>`;
        return `<td data-table-benchmark="${bench}" data-metric="${metric}"${metric === 0 ? ' class="benchmark-start"' : ""}><${tag} class="main-mean">${formatted}</${tag}>${std}</td>`;
      };
      let html = `<tr class="model-group-heading"><th colspan="10" scope="rowgroup"><span class="model-group-label">${escape(models[model])}<span class="model-method-count">${methods.length} methods</span></span></th></tr>`;
      for (const id of methods) {
        html += `<tr data-main-method="${id}" class="${modeInfo[id] ? `ours main-${id}` : ""}"><th scope="row">${methodName(id)}</th>`;
        for (const bench of Object.keys(benchmarks)) {
          const row = data[bench].find((r) => r[0] === id);
          for (let metric = 0; metric < 3; metric++) {
            html += metricCell(row, bench, metric);
          }
        }
        html += "</tr>";
      }
      $(`#main-results-${model}`).innerHTML = html;
      mobileHtml += `<section class="mobile-model-group" data-model="${model}" aria-labelledby="mobile-model-${model}"><h4 id="mobile-model-${model}">${escape(models[model])}<span class="model-method-count">${methods.length} methods</span></h4>`;
      for (const bench of Object.keys(benchmarks)) {
        mobileHtml += `<table class="mobile-benchmark-table" data-mobile-benchmark="${bench}"><caption>${escape(benchmarks[bench])}</caption><colgroup><col class="method-column"><col span="3"></colgroup><thead><tr><th scope="col">Method</th><th scope="col">P (%) <span>↑</span></th><th scope="col">C ($) <span>↓</span></th><th scope="col">T (s) <span>↓</span></th></tr></thead><tbody>`;
        for (const row of data[bench]) {
          const id = row[0];
          mobileHtml += `<tr data-main-method="${id}" class="${modeInfo[id] ? `ours main-${id}` : ""}"><th scope="row">${methodName(id)}</th>${[0, 1, 2].map((metric) => metricCell(row, bench, metric)).join("")}</tr>`;
        }
        mobileHtml += "</tbody></table>";
      }
      mobileHtml += "</section>";
    }
    $("#main-results-mobile").innerHTML = mobileHtml;
  }

  const runtime = {
    mode: "medium",
    profile: "standard",
    count: 0,
    timer: null,
  };
  const stages = {
    normal: [
      "Explore",
      "Work on the task with the selected reasoning effort. Gather the evidence needed for an effective solution.",
    ],
    economy: [
      "Economy",
      "Batch independent checks and avoid unchanged rereads. Keep the behavioral validation and final-state checks the task needs.",
    ],
    strict: [
      "Converge",
      "Stop broad exploration. Repair concrete failures, verify the requested behavior and final state, then finish.",
    ],
    blocked: [
      "Budget exhausted",
      "Further tool calls are blocked. Preserve the current files and give the best final answer supported by the evidence already collected.",
    ],
  };
  const policy = () => window.HARNESS_POLICIES[runtime.mode][runtime.profile];
  function stopPlayback() {
    clearInterval(runtime.timer);
    runtime.timer = null;
    $("#play-walkthrough").innerHTML =
      '<svg><use href="#i-play"/></svg><span>Play</span>';
    $("#play-walkthrough").setAttribute(
      "aria-label",
      "Play policy walkthrough",
    );
  }
  function renderRuntime() {
    const p = policy(),
      end = p.finalize_after_tools;
    const stage =
      runtime.count > end
        ? "blocked"
        : runtime.count >= p.strict_after_tools
          ? "strict"
          : runtime.count >= p.economy_after_tools
            ? "economy"
            : "normal";
    $("#tool-slider").max = end + 1;
    $("#tool-slider").value = runtime.count;
    $("#tool-slider").setAttribute(
      "aria-valuetext",
      `${runtime.count} tool-call attempts, ${stages[stage][0]}, limit ${end}`,
    );
    $("#tool-count").textContent = String(runtime.count).padStart(2, "0");
    $("#tool-limit").textContent = end;
    $("#reasoning-effort").textContent = p.reasoning_effort;
    $("#economy-value").textContent = p.economy_after_tools;
    $("#strict-value").textContent = p.strict_after_tools;
    $("#final-value").textContent = end;
    $("#economy-tick").textContent = `Economy · ${p.economy_after_tools}`;
    $("#strict-tick").textContent = `Converge · ${p.strict_after_tools}`;
    $("#final-tick").textContent = `Limit · ${end}`;
    $("#stage-name").textContent = stages[stage][0];
    $("#stage-description").textContent = stages[stage][1];
    $("#hook-output").dataset.stage = stage;
    $("#stage-dot").style.background =
      stage === "blocked"
        ? "#e6b48c"
        : stage === "strict"
          ? "#d2c1f3"
          : "#c4ceff";
    const e = (p.economy_after_tools / (end + 1)) * 100,
      s = (p.strict_after_tools / (end + 1)) * 100,
      f = (end / (end + 1)) * 100;
    $("#tool-slider").style.background =
      `linear-gradient(to right,#7189cd 0% ${e}%,#aab6e4 ${e}% ${s}%,#d6c190 ${s}% ${f}%,#c08b6d ${f}% 100%)`;
    $$("[data-runtime-mode]").forEach((b) =>
      b.setAttribute(
        "aria-pressed",
        String(b.dataset.runtimeMode === runtime.mode),
      ),
    );
  }
  $$("[data-runtime-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      stopPlayback();
      runtime.mode = b.dataset.runtimeMode;
      runtime.count = 0;
      renderRuntime();
    }),
  );
  $("#profile-select").addEventListener("change", (e) => {
    stopPlayback();
    runtime.profile = e.target.value;
    runtime.count = 0;
    renderRuntime();
  });
  $("#tool-slider").addEventListener("input", (e) => {
    stopPlayback();
    runtime.count = Number(e.target.value);
    renderRuntime();
  });
  $("#play-walkthrough").addEventListener("click", () => {
    if (runtime.timer) {
      stopPlayback();
      return;
    }
    if (runtime.count > policy().finalize_after_tools) runtime.count = 0;
    $("#play-walkthrough").innerHTML =
      '<svg><use href="#i-pause"/></svg><span>Pause</span>';
    $("#play-walkthrough").setAttribute(
      "aria-label",
      "Pause policy walkthrough",
    );
    runtime.timer = setInterval(() => {
      runtime.count++;
      renderRuntime();
      if (runtime.count > policy().finalize_after_tools) stopPlayback();
    }, 220);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPlayback();
  });

  const dialog = $("#figure-dialog");
  function showFigure(image, title) {
    const enlarged = dialog.querySelector("img");
    enlarged.src = image.getAttribute("src");
    enlarged.alt = image.alt;
    enlarged.width = image.width;
    enlarged.height = image.height;
    $("#dialog-title").textContent = title;
    dialog.showModal();
    dialog.scrollTo(0, 0);
    document.documentElement.classList.add("dialog-open");
  }
  $("#expand-figure").addEventListener("click", () => {
    showFigure($(".method-figure > img"), "The HarnessPro evolution loop");
  });
  $$("[data-enlarge-figure]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey)
        return;
      event.preventDefault();
      showFigure(
        link.querySelector("img"),
        link.closest("figure").querySelector("h3").textContent + " — evolution",
      );
    });
  });
  $("#close-figure").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () =>
    document.documentElement.classList.remove("dialog-open"),
  );
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) {
      const r = dialog.getBoundingClientRect();
      if (
        e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom
      )
        dialog.close();
    }
  });
  $("#copy-citation").addEventListener("click", async () => {
    const value = $("#bibtex").textContent,
      button = $("#copy-citation");
    let copied = false;
    try {
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(value);
        copied = true;
      }
    } catch {}
    if (!copied) {
      const area = document.createElement("textarea");
      area.value = value;
      area.style.cssText = "position:fixed;left:-9999px;top:0";
      document.body.append(area);
      area.select();
      try {
        copied = document.execCommand("copy");
      } catch {}
      area.remove();
      button.focus({ preventScroll: true });
    }
    $("#copy-status").textContent = copied
      ? "Citation copied to clipboard."
      : "Copy unavailable. Select and copy the citation below.";
    button.innerHTML = copied
      ? '<svg><use href="#i-check"/></svg><span>Copied</span>'
      : '<svg><use href="#i-copy"/></svg><span>Select citation below</span>';
    if (!copied) {
      const range = document.createRange();
      range.selectNodeContents($("#bibtex"));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    }
    setTimeout(() => {
      button.innerHTML =
        '<svg><use href="#i-copy"/></svg><span>Copy citation</span>';
    }, 2500);
  });
  const menu = $(".menu-toggle"),
    nav = $("#site-nav");
  function closeMenu() {
    menu.setAttribute("aria-expanded", "false");
    menu.setAttribute("aria-label", "Open navigation");
    nav.classList.remove("is-open");
  }
  menu.addEventListener("click", () => {
    const opened = menu.getAttribute("aria-expanded") === "true";
    menu.setAttribute("aria-expanded", String(!opened));
    menu.setAttribute(
      "aria-label",
      opened ? "Open navigation" : "Close navigation",
    );
    nav.classList.toggle("is-open", !opened);
  });
  $$("#site-nav a").forEach((a) => a.addEventListener("click", closeMenu));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && menu.getAttribute("aria-expanded") === "true") {
      closeMenu();
      menu.focus();
    }
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".site-header")) closeMenu();
  });
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          $$('#site-nav a[href^="#"]').forEach((a) => {
            if (a.getAttribute("href") === "#" + entry.target.id)
              a.setAttribute("aria-current", "location");
            else a.removeAttribute("aria-current");
          });
        }
      });
    },
    { rootMargin: "-15% 0px -60% 0px" },
  );
  ["overview", "method", "main-results", "evolution", "harnesses"].forEach(
    (id) => observer.observe($("#" + id)),
  );
  observer.observe($(".hero"));
  heroChart();
  renderMainTable();
  renderRuntime();

  const chartWidths = new Map();
  const chartObserver = new ResizeObserver((entries) => {
    for (const { target, contentRect } of entries) {
      const width = Math.round(contentRect.width);
      if (chartWidths.get(target) === width) continue;
      chartWidths.set(target, width);
      heroChart();
    }
  });
  chartObserver.observe($("#hero-chart"));
  document.fonts.ready.then(() => {
    heroChart();
  });
})();

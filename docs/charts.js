// Charts for the project page. Plain SVG, no dependencies.
// Every number here is copied from docs.md in this repository.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var css = getComputedStyle(document.documentElement);
  function tok(name, fb) { return css.getPropertyValue(name).trim() || fb; }
  var C = {
    gemm: tok("--series-1", "#1e2f4d"),
    attn: tok("--series-2", "#b8860b"),
    other: tok("--series-3", "#4f7cac"),
    grid: tok("--color-hairline", "rgba(15,23,42,0.12)"),
    surface: tok("--color-bg", "#f9f9f6"),
  };

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, attrs) {
    var t = el("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function tooltip(container) {
    var tip = document.createElement("div");
    tip.className = "tooltip";
    container.appendChild(tip);
    return {
      show: function (html, evt) {
        tip.innerHTML = html;
        tip.style.opacity = "1";
        var r = container.getBoundingClientRect();
        var x = evt.clientX - r.left + 14;
        var y = evt.clientY - r.top + 14;
        var w = tip.offsetWidth;
        if (x + w > r.width) x = Math.max(0, evt.clientX - r.left - w - 14);
        tip.style.left = x + "px";
        tip.style.top = y + "px";
      },
      hide: function () {
        tip.style.opacity = "0";
      },
    };
  }
  function legend(id, items) {
    var ul = document.getElementById(id);
    if (!ul) return;
    items.forEach(function (it) {
      var li = document.createElement("li");
      li.innerHTML = '<i style="background:' + it.color + '"></i>' + it.label;
      ul.appendChild(li);
    });
  }
  function fmt(v) {
    return v === null ? "<0.1%" : v.toFixed(1) + "%";
  }
  function sum(arr) {
    return arr.reduce(function (a, b) { return a + (b || 0); }, 0);
  }

  // ---------------------------------------------------------------- data
  // Nsight Systems kernel time share (docs.md, Milestone 3). null = "<0.1%".
  var S = [25, 50, 234];
  var KERNELS = [
    { name: "sgemm", group: "gemm", v: [88.8, 88.0, 83.7] },
    { name: "score", group: "attn", v: [7.2, 7.2, 10.2] },
    { name: "softmax", group: "attn", v: [1.5, 1.5, 1.0] },
    { name: "output", group: "attn", v: [1.2, 1.9, 3.2] },
    { name: "causal", group: "attn", v: [0.6, 0.6, 0.4] },
    { name: "rms_norm", group: "other", v: [0.3, 0.4, 0.3] },
    { name: "silu", group: "other", v: [0.1, 0.2, 0.7] },
    { name: "residual_add", group: "other", v: [0.1, 0.1, 0.4] },
    { name: "rope_apply", group: "other", v: [null, 0.1, 0.2] },
    { name: "embedding", group: "other", v: [null, null, null] },
    { name: "rope_precompute", group: "other", v: [null, null, null] },
  ];
  var GROUPS = [
    { key: "gemm", label: "SGEMM (all projections)", color: C.gemm },
    { key: "attn", label: "Attention (score, mask, softmax, output)", color: C.attn },
    { key: "other", label: "Everything else", color: C.other },
  ];

  // ---------------------------------------------------------------- table
  (function buildTable() {
    var t = document.getElementById("share-table");
    if (!t) return;
    var h = "<thead><tr><th>Kernel</th>" + S.map(function (s) {
      return '<th class="num">s = ' + s + "</th>";
    }).join("") + "</tr></thead><tbody>";
    KERNELS.forEach(function (k) {
      h += "<tr><td><code>" + k.name + "</code></td>" + k.v.map(function (v) {
        return '<td class="num">' + fmt(v) + "</td>";
      }).join("") + "</tr>";
    });
    t.innerHTML = h + "</tbody>";
  })();

  // ---------------------------------------------------------------- share chart
  legend("share-legend", GROUPS);
  function drawShare() {
    var box = document.getElementById("share-chart");
    if (!box) return;
    box.innerHTML = "";
    var W = Math.max(300, box.clientWidth);
    var narrow = W < 520;
    var padL = 64, padR = 8, padT = 6, padB = 28;
    var barH = 24, gap = 22;
    var H = padT + S.length * barH + (S.length - 1) * gap + padB;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, box);
    var tip = tooltip(box);
    var plotW = W - padL - padR;
    var x = function (p) { return padL + (p / 100) * plotW; };

    [0, 25, 50, 75, 100].forEach(function (p) {
      el("line", { x1: x(p), x2: x(p), y1: padT - 2, y2: H - padB + 4, stroke: C.grid, "stroke-width": 1 }, svg);
      txt(svg, x(p), H - 8, p + "%", { "text-anchor": p === 0 ? "start" : p === 100 ? "end" : "middle" });
    });

    S.forEach(function (s, i) {
      var y = padT + i * (barH + gap);
      txt(svg, padL - 10, y + barH / 2 + 4, "s = " + s, { "text-anchor": "end", class: "val" });
      var acc = 0;
      GROUPS.forEach(function (g, gi) {
        var members = KERNELS.filter(function (k) { return k.group === g.key; });
        var val = sum(members.map(function (k) { return k.v[i]; }));
        var x0 = x(acc), x1 = x(acc + val);
        acc += val;
        var last = gi === GROUPS.length - 1;
        var w = Math.max(1, x1 - x0 - (last ? 0 : 2)); // 2px surface gap
        var r = last ? 4 : 0;
        var path;
        if (r) {
          var rr = Math.min(r, w / 2);
          path = "M" + x0 + "," + y + " h" + (w - rr) + " a" + rr + "," + rr + " 0 0 1 " + rr + "," + rr +
            " v" + (barH - 2 * rr) + " a" + rr + "," + rr + " 0 0 1 " + -rr + "," + rr + " h" + -(w - rr) + " z";
        } else {
          path = "M" + x0 + "," + y + " h" + w + " v" + barH + " h" + -w + " z";
        }
        var p = el("path", { d: path, fill: g.color }, svg);
        var detail = members.map(function (k) {
          return "<div>" + k.name + ": " + fmt(k.v[i]) + "</div>";
        }).join("");
        var html = '<b><span class="sw" style="background:' + g.color + '"></span>' + g.label +
          "</b><div>s = " + s + " &middot; " + val.toFixed(1) + "% of GPU time</div>" +
          (members.length > 1 ? '<div class="dim" style="margin-top:4px">' + detail + "</div>" : "");
        // generous hit target
        var hit = el("rect", { x: x0, y: y - 6, width: Math.max(10, x1 - x0), height: barH + 12, fill: "transparent" }, svg);
        [p, hit].forEach(function (n) {
          n.addEventListener("mousemove", function (e) { tip.show(html, e); });
          n.addEventListener("mouseleave", tip.hide);
        });
        // inline label only where it fits
        var label = val.toFixed(1) + "%";
        var need = label.length * 7 + 14;
        if (gi === 0 && w > need) {
          txt(svg, x0 + 8, y + barH / 2 + 4, label, { fill: "#fff", style: "fill:#fff;font-weight:500", "pointer-events": "none" });
        } else if (gi === 1 && w > need && !narrow) {
          txt(svg, x0 + w / 2, y + barH / 2 + 4, label, { "text-anchor": "middle", style: "fill:#0f172a;font-weight:500", "pointer-events": "none" });
        }
      });
    });
  }

  // ---------------------------------------------------------------- roofline
  var POINTS = [
    { name: "RMSNorm", ai: 0.345, gflops: 108, color: C.attn, note: "629.66 µs · ~314 GB/s" },
    { name: "SGEMM (register-tiled)", ai: 30.05, gflops: 4960, color: C.gemm, note: "27.73 ms · ~165 GB/s" },
  ];
  legend("roof-legend", POINTS.map(function (p) { return { label: p.name, color: p.color }; }));
  function drawRoof() {
    var box = document.getElementById("roof-chart");
    if (!box) return;
    box.innerHTML = "";
    var W = Math.max(300, box.clientWidth);
    var H = Math.round(Math.min(340, Math.max(240, W * 0.5)));
    var padL = 58, padR = 14, padT = 12, padB = 42;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, box);
    var tip = tooltip(box);
    var lx0 = Math.log10(0.1), lx1 = Math.log10(100);
    var ly0 = Math.log10(10), ly1 = Math.log10(10000);
    var x = function (v) { return padL + ((Math.log10(v) - lx0) / (lx1 - lx0)) * (W - padL - padR); };
    var y = function (v) { return H - padB - ((Math.log10(v) - ly0) / (ly1 - ly0)) * (H - padT - padB); };

    [0.1, 1, 10, 100].forEach(function (v) {
      el("line", { x1: x(v), x2: x(v), y1: padT, y2: H - padB, stroke: C.grid }, svg);
      txt(svg, x(v), H - padB + 18, String(v), { "text-anchor": v === 0.1 ? "start" : v === 100 ? "end" : "middle" });
    });
    [[10, "10 G"], [100, "100 G"], [1000, "1 T"], [10000, "10 T"]].forEach(function (t) {
      el("line", { x1: padL, x2: W - padR, y1: y(t[0]), y2: y(t[0]), stroke: C.grid }, svg);
      txt(svg, padL - 8, y(t[0]) + 4, t[1], { "text-anchor": "end" });
    });
    txt(svg, padL + (W - padL - padR) / 2, H - 6, "Arithmetic intensity (FLOP / byte)", { "text-anchor": "middle" });
    txt(svg, 0, 0, "FLOP / s", { "text-anchor": "middle", transform: "translate(12," + (padT + (H - padT - padB) / 2) + ") rotate(-90)" });

    POINTS.forEach(function (p) {
      var cx = x(p.ai), cy = y(p.gflops);
      var dot = el("circle", { cx: cx, cy: cy, r: 6, fill: p.color, stroke: C.surface, "stroke-width": 2 }, svg);
      var hit = el("circle", { cx: cx, cy: cy, r: 16, fill: "transparent" }, svg);
      var perf = p.gflops >= 1000 ? (p.gflops / 1000).toFixed(2) + " TFLOP/s" : p.gflops + " GFLOP/s";
      var html = '<b><span class="sw" style="background:' + p.color + '"></span>' + p.name + "</b><div>" +
        p.ai + " FLOP/B &middot; " + perf + '</div><div class="dim">' + p.note + "</div>";
      [dot, hit].forEach(function (n) {
        n.addEventListener("mousemove", function (e) { tip.show(html, e); });
        n.addEventListener("mouseleave", tip.hide);
      });
      var right = cx < W * 0.6;
      txt(svg, cx + (right ? 12 : -12), cy - 10, p.name.split(" ")[0] + " · " + perf, {
        "text-anchor": right ? "start" : "end", class: "val",
      });
    });
  }

  function drawAll() { drawShare(); drawRoof(); }
  var t;
  window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(drawAll, 120); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawAll);
  drawAll();
})();

/* Opening layer on top of the hero: 25 annotated clips drop in row by row and form a 5 x 5 square, one row per question
   (Which tool, What action, When, How, Where). Scrolling gathers each row into a small deck
   that flies to its own card in the hero, and the hero (paper name, kitten) appears. */
(function () {
  const canvas = document.getElementById("scene"), ctx = canvas.getContext("2d");
  const v = document.getElementById("vMosaic"), poster = new Image(); poster.src = "assets/opening_poster.jpg";
  const opening = document.getElementById("opening"), slot = document.getElementById("mosaicSlot"), hint = document.getElementById("scrollHint"), card = document.getElementById("openCard");
  const cards = [].slice.call(slot.querySelectorAll(".q-card i"));
  const reveal = [".nav", ".hero__title", ".hero__paper", ".bubble", ".hero__body", ".mosaic-slot", ".paw-menu"].map((s) => document.querySelector(s)).filter(Boolean);
  const PAPER = getComputedStyle(document.documentElement).getPropertyValue("--paper").trim() || "#e7e7e7", INK = "#050505", ORANGE = "#C4530B";
  const ROWS = ["Which tool", "What action", "When", "How", "Where"];
  const G = 5, N = G * G;
  const q = new URLSearchParams(location.search);                       // ?p=0.5&t=9 freezes the scene (for screenshots)
  const forceP = q.has("p") ? parseFloat(q.get("p")) : null, forceT = q.has("t") ? parseFloat(q.get("t")) : null;
  if (q.has("fan")) slot.classList.add("fan");                          // ?fan : the five cards held like a hand of cards instead of a row
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (document.fonts && document.fonts.load) document.fonts.load('600 14px "DynaPuff"').catch(() => {});

  const tiles = [];
  for (let i = 0; i < N; i++) { const r = Math.floor(i / G), c = i % G; tiles.push({ r, c, delay: 0.2 + r * 0.26 + c * 0.06 }); }   // row after row, left to right
  tiles.sort((a, b) => b.c - a.c);                                       // the first clip of a row is drawn last: it is the one the card keeps
  const LAND = 0.75, READY = 0.2 + 4 * 0.26 + 4 * 0.06 + LAND;           // when the whole square has landed

  let W = 0, H = 0, dpr = 1;
  function resize() { dpr = Math.min(devicePixelRatio || 1, 2); W = canvas.clientWidth; H = canvas.clientHeight; canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
  addEventListener("resize", resize); resize();
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x)), seg = (p, a, b) => clamp((p - a) / (b - a));
  const ease = (x) => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2, lerp = (a, b, t) => a + (b - a) * t;
  const outCubic = (x) => 1 - Math.pow(1 - x, 3), outBack = (x) => { const c1 = 1.6, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };
  function pill(x, y, w, h, fill, a) { ctx.globalAlpha = a; ctx.fillStyle = fill; ctx.beginPath(); ctx.roundRect(x, y, w, h, h * 0.34); ctx.fill(); }

  const t0 = performance.now();
  function frame(now) {
    const t = forceT !== null ? forceT : reduce ? 99 : (now - t0) / 1000;
    const p = forceP !== null ? forceP : clamp(-opening.getBoundingClientRect().top / (opening.offsetHeight - innerHeight));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const src = v.readyState >= 2 ? v : poster, mw = src.videoWidth || src.naturalWidth;
    if (!mw) { requestAnimationFrame(frame); return; }
    const ts = mw / G, u = clamp(Math.min(W / 982, H / 696), 0.8, 1.7);

    // the square under the title card
    const cr = canvas.getBoundingClientRect(), big = Math.min(W * 0.9, H * 0.69), x0 = (W - big) / 2, y0 = H * 0.215, cell = big / G;
    const dst = cards.map((el) => { const r = el.getBoundingClientRect(), m = new DOMMatrixReadOnly(getComputedStyle(el.parentNode).transform);
      return { x: r.left - cr.left + r.width / 2, y: r.top - cr.top + r.height / 2, s: el.offsetWidth, rot: Math.atan2(m.b, m.a) }; });

    const veil = 1 - seg(p, 0.12, 0.6);                                   // the page colour hides the hero until the rows start to leave
    if (veil > 0) { ctx.globalAlpha = veil; ctx.fillStyle = PAPER; ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1; }

    const still = 1 - seg(p, 0.02, 0.1), act = -1, glow = 0;             // (the rows no longer light up in turn: the square simply stands)

    // the name of each row, left of the square, while it is still a square
    if (still > 0 && x0 > 130 * u) {
      ctx.font = `600 ${14 * u}px "DynaPuff", sans-serif`; ctx.textBaseline = "middle"; ctx.textAlign = "center";
      ROWS.forEach((name, r) => {
        const a = outBack(clamp((t - (0.2 + r * 0.26) - 0.25) / 0.45)) * still; if (a <= 0) return;
        const g = r === act ? glow : 0, w = ctx.measureText(name).width + 24 * u, h = 30 * u, x = x0 - 16 * u - w - (1 - Math.min(a, 1)) * 18 * u - g * 5 * u, y = y0 + r * cell + cell / 2 - h / 2;
        pill(x, y, w, h, INK, Math.min(a, 1)); if (g > 0) pill(x, y, w, h, ORANGE, g);
        ctx.globalAlpha = Math.min(a, 1); ctx.fillStyle = "#fff"; ctx.fillText(name, x + w / 2, y + h / 2 + 0.5);
      });
      ctx.globalAlpha = 1;
    }

    for (const tl of tiles) {
      const l = clamp((t - tl.delay) / LAND); if (l <= 0) continue;
      const f = seg(p, 0.05 + tl.r * 0.04, 0.64 + tl.r * 0.04), gather = outCubic(seg(f, 0, 0.42)), fly = ease(seg(f, 0.3, 1)), d = dst[tl.r];
      const g = tl.r === act ? glow : 0, dim = act < 0 ? 1 : 1 - 0.3 * glow * (tl.r === act ? 0 : 1);
      // at rest in the square -> gathered into a small deck at the middle of its row -> along an arc to the row's card
      const rx = x0 + tl.c * cell + cell / 2, ry = y0 + tl.r * cell + cell / 2, deck = (tl.c - 2) * cell * 0.17;
      const x = lerp(lerp(rx, x0 + big / 2 + deck, gather), d.x, fly), y = lerp(ry, d.y, fly) - Math.sin(Math.PI * fly) * H * 0.07 - (1 - outBack(l)) * H * 0.16;
      const size = lerp(cell * 0.94, d.s, fly) * lerp(0.7, 1, outBack(l)) * (1 + 0.05 * g) * (1 + 0.1 * Math.sin(Math.PI * seg(f, 0.88, 1)));
      const alpha = Math.min(1, l * 2.5) * dim * (tl.c ? 1 - seg(fly, 0.7, 0.98) : 1), rot = (tl.c - 2) * 0.09 * gather * (1 - fly) + d.rot * fly;
      if (alpha <= 0) continue;
      const round = lerp(0.1, 0, fly);
      ctx.save(); ctx.globalAlpha = alpha; ctx.translate(x, y); ctx.rotate(rot);
      if (gather > 0 && fly < 1) { ctx.shadowColor = "rgba(0,0,0,.22)"; ctx.shadowBlur = 14 * u * gather * (1 - fly); ctx.shadowOffsetY = 4 * u * gather * (1 - fly); ctx.fillStyle = PAPER; ctx.beginPath(); ctx.roundRect(-size / 2, -size / 2, size, size, size * round); ctx.fill(); ctx.shadowColor = "transparent"; }
      if (round > 0.005) { ctx.beginPath(); ctx.roundRect(-size / 2, -size / 2, size, size, size * round); ctx.clip(); }
      ctx.drawImage(src, tl.c * ts, tl.r * ts, ts, ts, -size / 2, -size / 2, size, size); ctx.restore();
    }

    const show = seg(p, 0.35, 0.8);
    for (const el of reveal) el.style.opacity = String(show);
    hint.style.opacity = String((1 - seg(p, 0.02, 0.1)) * clamp((t - READY) / 0.6));
    card.style.opacity = String((1 - seg(p, 0.04, 0.2)) * clamp((t - 0.1) / 0.6));
    requestAnimationFrame(frame);
  }
  v.play().catch(() => {});
  requestAnimationFrame(frame);
})();

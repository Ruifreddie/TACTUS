/* Opening layer on top of the hero: the dataset's clips fly in and cover the screen as one square mosaic;
   scrolling shrinks the square into its slot beside the title, and the hero (paper name, kitten) appears. */
(function () {
  const canvas = document.getElementById("scene"), ctx = canvas.getContext("2d");
  const v = document.getElementById("vMosaic"), poster = new Image(); poster.src = "assets/mosaic_poster.jpg";
  const opening = document.getElementById("opening"), slot = document.getElementById("mosaicSlot"), hint = document.getElementById("scrollHint"), card = document.getElementById("openCard");
  const reveal = [".nav", ".hero__title", ".hero__paper", ".bubble", ".hero__body", ".mosaic-slot", ".paw-menu"].map((s) => document.querySelector(s)).filter(Boolean);
  const PAPER = getComputedStyle(document.documentElement).getPropertyValue("--paper").trim() || "#e7e7e7";
  const G = 12, N = G * G;
  const q = new URLSearchParams(location.search);                       // ?p=0.5&t=9 freezes the scene (for screenshots)
  const forceP = q.has("p") ? parseFloat(q.get("p")) : null, forceT = q.has("t") ? parseFloat(q.get("t")) : null;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const tiles = [];
  for (let i = 0; i < N; i++) {
    const r = Math.floor(i / G), c = i % G, d = Math.hypot(r - (G - 1) / 2, c - (G - 1) / 2) / (G * 0.707);
    tiles.push({ r, c, a: Math.atan2(r - (G - 1) / 2, c - (G - 1) / 2) + (rnd() - 0.5) * 1.6, dist: 0.75 + rnd() * 0.5, rot: (rnd() - 0.5) * 2.6, delay: 0.1 + d * 0.8 + rnd() * 0.25 });
  }
  const LAND = 1.1;

  let W = 0, H = 0, dpr = 1;
  function resize() { dpr = Math.min(devicePixelRatio || 1, 2); W = canvas.clientWidth; H = canvas.clientHeight; canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
  addEventListener("resize", resize); resize();
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x)), seg = (p, a, b) => clamp((p - a) / (b - a));
  const ease = (x) => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2, lerp = (a, b, t) => a + (b - a) * t;
  const outBack = (x) => { const c1 = 1.2, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };

  const t0 = performance.now();
  function frame(now) {
    const t = forceT !== null ? forceT : reduce ? 99 : (now - t0) / 1000;
    const p = forceP !== null ? forceP : clamp(-opening.getBoundingClientRect().top / (opening.offsetHeight - innerHeight));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const src = v.readyState >= 2 ? v : poster, mw = src.videoWidth || src.naturalWidth;
    if (!mw) { requestAnimationFrame(frame); return; }
    const ts = mw / G, shrink = ease(seg(p, 0.05, 0.8));

    // the square: from covering the screen to its slot in the hero
    const sr = slot.getBoundingClientRect(), cr = canvas.getBoundingClientRect();
    const big = Math.max(W, H) * 1.04, S = lerp(big, sr.width, shrink);
    const x0 = lerp((W - big) / 2, sr.left - cr.left, shrink), y0 = lerp((H - big) / 2, sr.top - cr.top, shrink), cell = S / G;

    const veil = 1 - seg(p, 0.12, 0.6);                                   // the page colour hides the hero until the square starts to shrink
    if (veil > 0) { ctx.globalAlpha = veil; ctx.fillStyle = PAPER; ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1; }

    const gapk = lerp(0.94, 1, shrink), round = lerp(0.14, 0, shrink);   // tiles close up into one pixel-sharp picture as they land in the slot
    for (const tl of tiles) {
      const l = clamp((t - tl.delay) / LAND); if (l <= 0) continue; const land = outBack(l);
      const hx = x0 + tl.c * cell + cell / 2, hy = y0 + tl.r * cell + cell / 2, R = Math.hypot(W, H) * tl.dist;
      const x = lerp(W / 2 + Math.cos(tl.a) * R, hx, land), y = lerp(H / 2 + Math.sin(tl.a) * R, hy, land), size = cell * gapk * lerp(0.4, 1, l) + (shrink > 0.98 ? 0.6 : 0);
      if (x < -size || y < -size || x > W + size || y > H + size) continue;
      ctx.save(); ctx.globalAlpha = Math.min(1, l * 2); ctx.translate(x, y); ctx.rotate(tl.rot * (1 - land));
      if (round > 0.01) { ctx.beginPath(); ctx.roundRect(-size / 2, -size / 2, size, size, size * round); ctx.clip(); }
      ctx.drawImage(src, tl.c * ts, tl.r * ts, ts, ts, -size / 2, -size / 2, size, size); ctx.restore();
    }

    const show = seg(p, 0.35, 0.8);
    for (const el of reveal) el.style.opacity = String(show);
    hint.style.opacity = String((1 - seg(p, 0.02, 0.1)) * clamp((t - 2.0) / 0.6));
    card.style.opacity = String((1 - seg(p, 0.04, 0.2)) * clamp((t - 1.9) / 0.7));
    requestAnimationFrame(frame);
  }
  v.play().catch(() => {});
  requestAnimationFrame(frame);
})();

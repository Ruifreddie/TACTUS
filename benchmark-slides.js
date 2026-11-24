(() => {
  const story = document.getElementById('benchmark-distribution');
  const pin = story.querySelector('.split-story__pin');
  const viewport = story.querySelector('.split-story__viewport');
  const nav = story.querySelector('.split-story__nav');
  const buttons = [...nav.querySelectorAll('button')];
  const slides = [...story.querySelector('.split-story__track').children];
  const desktop = matchMedia('(min-width: 900px) and (min-height: 640px)');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let pinned = false;
  let travel = 0;
  let top = 0;
  let active = -1;
  let pending = false;

  // S9 provides split sizes. Both panels use the same 58 x 28 grid.
  for (const canvas of story.querySelectorAll('.split-overview canvas')) {
    const count = Number(canvas.parentElement.querySelector('b').textContent.replaceAll(',', ''));
    const ctx = canvas.getContext('2d');
    if (!ctx) continue;
    function paint() {
      const width = Math.round(canvas.clientWidth * Math.min(2, devicePixelRatio || 1));
      if (!width) return;
      const step = width / 58;
      canvas.width = width;
      canvas.height = Math.round(step * 28);
      ctx.fillStyle = canvas.dataset.color;
      ctx.beginPath();
      for (let i = 0; i < count; i++) {
        const x = (Math.floor(i / 28) + 0.5) * step;
        const y = (i % 28 + 0.5) * step;
        ctx.moveTo(x + step * 0.34, y);
        ctx.arc(x, y, step * 0.34, 0, Math.PI * 2);
      }
      ctx.fill();
    }
    new ResizeObserver(paint).observe(canvas);
  }

  function markActive(index) {
    if (index === active) return;
    active = index;
    buttons.forEach((button, i) => {
      button.setAttribute('aria-pressed', String(i === index));
      slides[i].inert = i !== index;
    });
  }

  function sync() {
    pending = false;
    if (pinned) {
      const progress = (top - story.getBoundingClientRect().top) / travel;
      const slide = Math.max(0, Math.min(1, (progress - 0.14) / 0.72));
      viewport.scrollLeft = viewport.clientWidth * slide;
    }
    markActive(viewport.scrollLeft < viewport.clientWidth * 0.5 ? 0 : 1);
  }

  function schedule() {
    if (!pending) { pending = true; requestAnimationFrame(sync); }
  }

  function layout() {
    const height = pin.offsetHeight;
    top = Math.round(Math.max(12, (innerHeight - height) / 2));
    travel = Math.round(innerHeight * 0.72);
    pinned = desktop.matches && !reduced.matches && height + 48 <= innerHeight;
    story.style.setProperty('--pin-top', top + 'px');
    story.style.setProperty('--panel-height', height + 'px');
    story.style.setProperty('--slide-travel', travel + 'px');
    story.classList.toggle('is-pinned', pinned);
    schedule();
  }

  function go(index, instant = false) {
    const behavior = instant || reduced.matches ? 'instant' : 'smooth';
    if (pinned) {
      const y = scrollY + story.getBoundingClientRect().top - top + index * travel;
      window.scrollTo({ top: y, behavior });
    } else {
      viewport.scrollTo({ left: index * viewport.clientWidth, behavior });
    }
  }

  nav.hidden = false;
  story.classList.add('is-ready');
  buttons.forEach((button, index) => button.addEventListener('click', () => go(index)));
  viewport.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('scroll', () => { if (pinned) schedule(); }, { passive: true });
  window.addEventListener('resize', layout);
  reduced.addEventListener('change', layout);
  new ResizeObserver(layout).observe(pin);
  layout();
  function followAnchor() {
    if (location.hash === '#gold-test-set') go(1, true);
    if (location.hash === '#benchmark-distribution' || location.hash === '#held-out-splits') go(0, true);
  }
  window.addEventListener('hashchange', followAnchor);
  window.addEventListener('load', followAnchor, { once: true });
  followAnchor();
})();

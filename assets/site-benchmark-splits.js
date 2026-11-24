(() => {
  const story = document.querySelector('.bs-story');
  if (!story) return;
  const card = story.querySelector('.bs-card');
  const stat = card.querySelector('.bs-stat');
  const label = card.querySelector('.bs-stat-label');
  const caption = card.querySelector('#bs-chart-caption');
  const groups = [...story.querySelectorAll('.bs-item')].map((item) => ({
    item,
    button: item.matches('button') ? item : item.querySelector('button'),
    name: item.dataset.name,
    clips: Number(item.dataset.clips),
    videos: Number(item.dataset.videos)
  }));
  const clipTotal = groups.reduce((sum, group) => sum + group.clips, 0);
  const videoTotal = groups.reduce((sum, group) => sum + group.videos, 0);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const fullCap = 3.082;
  const format = new Intl.NumberFormat('en-US');
  let active = -1;
  let painted;
  let tween = 0;

  function paint(values) {
    painted = values;
    card.style.setProperty('--clip-top', values.clipTop + '%');
    card.style.setProperty('--video-top', values.videoTop + '%');
    card.style.setProperty('--cap-h', values.cap + '%');
    card.style.setProperty('--label-op', String(values.opacity));
    stat.textContent = values.share.toFixed(2) + '%';
  }

  function select(index, instantly = false) {
    const changed = active !== index;
    if (!changed && !instantly) return;
    active = index;
    groups.forEach(({ item, button }, i) => {
      item.classList.toggle('is-open', i === index);
      button.setAttribute('aria-expanded', String(i === index));
    });

    const group = groups[index];
    const share = group.clips / clipTotal * 100;
    const videoShare = group.videos / videoTotal * 100;
    const target = { clipTop: 100 - share, videoTop: 100 - videoShare, share, cap: fullCap, opacity: 1 };
    const text = group.name + ' · share of clips';
    caption.textContent = group.name + ': ' + format.format(group.clips) + ' of ' + format.format(clipTotal)
      + ' clips (' + share.toFixed(2) + '%), and ' + group.videos + ' of ' + videoTotal
      + ' videos (' + videoShare.toFixed(2) + '%).';

    if (changed) {
      const split = group.name.toLowerCase();
      story.dataset.activeSplit = split;
      document.dispatchEvent(new CustomEvent('tactus:split-change', {
        detail: { split, name: group.name, clips: group.clips, videos: group.videos, clipTotal, videoTotal }
      }));
    }

    cancelAnimationFrame(tween);
    if (instantly || reduced.matches || !painted) {
      paint(target);
      label.textContent = text;
      return;
    }
    const from = painted;
    const start = performance.now();
    const out = (elapsed, duration, exponent) => 1 - Math.pow(1 - Math.max(0, Math.min(1, elapsed / duration)), exponent);
    const mix = (a, b, progress) => a + (b - a) * progress;
    function tick(now) {
      const t = now - start;
      const bar = out(t, 370, 3);
      if (t >= 160) label.textContent = text;
      paint({
        clipTop: mix(from.clipTop, target.clipTop, bar),
        videoTop: mix(from.videoTop, target.videoTop, bar),
        share: mix(from.share, share, out(t, 800, 2)),
        cap: t < 190 ? from.cap * (1 - out(t, 190, 2)) : fullCap * out(t - 390, 250, 3),
        opacity: t < 160 ? from.opacity * (1 - out(t, 160, 2)) : out(t - 230, 200, 2)
      });
      if (t < 800) tween = requestAnimationFrame(tick);
      else paint(target);
    }
    tween = requestAnimationFrame(tick);
  }

  groups.forEach(({ button }, index) => {
    button.addEventListener('click', () => select(index));
    button.addEventListener('keydown', (event) => {
      let next;
      switch (event.key) {
        case 'ArrowDown': case 'ArrowRight': next = (index + 1) % groups.length; break;
        case 'ArrowUp': case 'ArrowLeft': next = (index + groups.length - 1) % groups.length; break;
        case 'Home': next = 0; break;
        case 'End': next = groups.length - 1; break;
        default: return;
      }
      event.preventDefault();
      groups[next].button.focus({ preventScroll: true });
      select(next);
    });
  });
  reduced.addEventListener('change', () => select(active, true));
  select(0, true);

  if (!reduced.matches && 'IntersectionObserver' in window && Element.prototype.animate) {
    const stage = story.querySelector('.bs-stage');
    const entrance = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      entrance.disconnect();
      if (reduced.matches) return;
      const reveal = (selector, from, duration, delay) =>
        stage.querySelectorAll(selector).forEach((node) => node.animate(
          [from, { opacity: 1, transform: 'none', clipPath: 'inset(0)' }],
          { duration, delay, easing: 'cubic-bezier(.16,1,.3,1)', fill: 'backwards' }
        ));
      reveal('.bs-headline-line', { opacity: 0, transform: 'translateY(12px)' }, 540, 30);
      reveal('.bs-chart-grid', { opacity: 0, clipPath: 'inset(0 100% 0 0)' }, 560, 80);
      reveal('.bs-bar-videos', { transform: 'scaleY(0)' }, 640, 100);
      reveal('.bs-bar-clips', { transform: 'scaleY(0)' }, 640, 155);
      reveal('.bs-bar-cap', { opacity: 0 }, 330, 220);
    }, { threshold: 0.2 });
    entrance.observe(stage);
  }
})();

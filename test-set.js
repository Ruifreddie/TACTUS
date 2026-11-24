(() => {
  const figure = document.getElementById('gold-test-set');
  const canvas = figure.querySelector('canvas');
  const context = canvas.getContext('2d');
  if (!context) return;

  const table = figure.querySelector('table');
  const body = table.tBodies[0];
  const controls = figure.querySelector('.test-set__controls');
  const pairGroups = Array.from(body.rows, (row) => ({
    name: row.cells[0].textContent,
    tool: row.dataset.tool,
    count: Number(row.cells[1].textContent.replaceAll(',', '')),
    color: row.style.getPropertyValue('--dot')
  }));
  // Pairs from the same tool are adjacent, so each dot keeps its position in both views.
  const toolGroups = [];
  for (const pair of pairGroups) {
    const tool = toolGroups.find((group) => group.name === pair.tool);
    if (tool) tool.count += pair.count;
    else toolGroups.push({ name: pair.tool, count: pair.count, color: pair.color });
  }
  const total = pairGroups.reduce((sum, group) => sum + group.count, 0);
  const rows = 28;
  const columns = Math.ceil(total / rows);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let groups = pairGroups;
  let selection = null;
  let frame = 0;
  let cell = 0;

  function paint(palette, reveal = 1) {
    let point = 0;
    for (const group of palette) {
      context.fillStyle = selection && group !== selection ? '#484a44' : group.color;
      context.beginPath();
      for (let i = 0; i < group.count; i++, point++) {
        const column = Math.floor(point / rows);
        const x = (column + 0.5) * cell;
        const y = (point % rows + 0.5) * cell;
        const wave = Math.min(1, Math.max(0, 2 * reveal - column / columns));
        const radius = cell * (0.13 + 0.21 * wave);
        context.moveTo(x + radius, y);
        context.arc(x, y, radius, 0, 2 * Math.PI);
      }
      context.fill();
    }
  }

  function draw(progress = 1, previous = null) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (previous && progress < 1) {
      paint(previous);
      context.globalAlpha = progress;
    }
    paint(groups, previous ? 1 : progress);
    context.globalAlpha = 1;
  }

  function animate(duration, previous = null) {
    cancelAnimationFrame(frame);
    if (reducedMotion.matches) { draw(); return; }
    const start = performance.now();
    function tick(now) {
      const progress = Math.min(1, (now - start) / duration);
      draw(progress, previous);
      if (progress < 1) frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
  }

  function resize() {
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth * scale);
    if (!width || width === canvas.width && cell) return;
    cell = width / columns;
    canvas.width = width;
    canvas.height = Math.round(cell * rows);
    draw();
  }

  function highlight(group) {
    cancelAnimationFrame(frame);
    selection = selection === group ? null : group;
    groups.forEach((item, i) => {
      body.rows[i].toggleAttribute('data-active', selection === item);
      body.rows[i].querySelector('button').setAttribute('aria-pressed', String(selection === item));
    });
    draw();
  }

  function showLegend() {
    body.replaceChildren();
    for (const group of groups) {
      const row = body.insertRow();
      row.style.setProperty('--dot', group.color);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'test-set__name';
      button.textContent = group.name;
      button.setAttribute('aria-label', 'Highlight ' + group.name + ' clips');
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => highlight(group));
      row.insertCell().append(button);
      row.insertCell().textContent = group.count.toLocaleString('en-US');
      row.insertCell().textContent = (100 * group.count / total).toFixed(1) + '%';
    }
  }

  for (const button of controls.querySelectorAll('button')) {
    button.addEventListener('click', () => {
      const byTool = button.dataset.view === 'tool';
      const next = byTool ? toolGroups : pairGroups;
      if (next === groups) return;
      const previous = groups;
      groups = next;
      selection = null;
      for (const other of controls.querySelectorAll('button')) {
        other.setAttribute('aria-pressed', String(other === button));
      }
      table.tHead.rows[0].cells[0].textContent = byTool ? 'Tool' : 'Tool–action pair';
      table.setAttribute('aria-label', 'Gold test clips by ' + (byTool ? 'tool' : 'tool–action pair'));
      showLegend();
      animate(340, previous);
    });
  }
  controls.hidden = false;
  figure.querySelector('.test-set__plot').hidden = false;
  showLegend();
  resize();
  new ResizeObserver(resize).observe(canvas);
  const entry = new IntersectionObserver((entries) => {
    if (entries.some((item) => item.isIntersecting)) {
      entry.disconnect();
      animate(800);
    }
  }, { threshold: 0.25 });
  entry.observe(canvas);
  reducedMotion.addEventListener('change', () => { cancelAnimationFrame(frame); draw(); });
})();

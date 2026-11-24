(() => {
  const figure = document.getElementById('gold-test-set');
  if (!figure) return;
  const canvas = figure.querySelector('canvas');
  const context = canvas.getContext('2d');
  const summary = document.getElementById('benchmark-distribution');
  const table = figure.querySelector('#test-pair-counts');
  const body = table.tBodies[0];
  const controls = figure.querySelector('#composition-mode');
  const tabs = Array.from(controls.querySelectorAll('[data-view]'));
  const panel = figure.querySelector('#composition-panel');
  const testBody = figure.querySelector('.test-set__body');
  const coverage = figure.querySelector('#split-class-coverage');
  const sources = figure.querySelector('#split-sources');
  const title = figure.querySelector('#test-set-title');
  const splitNames = { train: 'training', val: 'validation', test: 'test' };
  const sourceNames = ['Cholec80', 'SurgVU', 'YouTube', 'HeiChole', 'CRCD'];
  const sourceVideos = { train: [24, 26, 15, 8, 6], val: [3, 3, 3, 1, 1], test: [3, 3, 3, 1, 1] };
  let split = Object.hasOwn(splitNames, summary?.dataset.activeSplit) ? summary.dataset.activeSplit : 'train';
  let view = 'pair';
  const pairGroups = Array.from(body.rows, (row) => ({
    name: row.cells[0].textContent.trim(),
    action: row.cells[0].textContent.trim().split('–')[1],
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
    if (!context || !cell || testBody.hidden) return;
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
      button.addEventListener('click', () => { if (!testBody.hidden && groups.includes(group)) highlight(group); });
      row.insertCell().append(button);
      row.insertCell().textContent = group.count.toLocaleString('en-US');
      row.insertCell().textContent = (100 * group.count / total).toFixed(1) + '%';
    }
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function renderMatrix() {
    const matrix = element('table', 'sc-matrix');
    matrix.createCaption().textContent = 'Six annotated tool–action pairs';
    const actions = ['Dissect', 'Grasp', 'Cut', 'Clip'];
    const head = matrix.createTHead().insertRow();
    for (const name of ['Instrument', ...actions]) {
      const header = element('th', '', name);
      header.scope = 'col';
      head.append(header);
    }
    const rows = matrix.createTBody();
    for (const tool of toolGroups) {
      const row = rows.insertRow();
      const header = element('th', '', tool.name);
      header.scope = 'row';
      row.append(header);
      for (const action of actions) {
        const pair = pairGroups.find((item) => item.tool === tool.name && item.action === action.toLowerCase());
        const cell = row.insertCell();
        const mark = element('span', pair ? 'sc-dot' : 'sc-dash', pair ? '' : '–');
        mark.setAttribute('aria-hidden', 'true');
        if (pair) mark.style.setProperty('--dot', pair.color);
        cell.append(mark, element('span', 'sr-only', pair ? 'Included pair' : 'Outside the six annotated pairs'));
      }
    }
    coverage.replaceChildren(matrix);
  }

  function renderTools() {
    const list = element('dl', 'sc-tool-list');
    for (const tool of toolGroups) {
      const row = element('div', 'sc-tool-row');
      const actions = element('dd', 'sc-actions');
      for (const pair of pairGroups.filter((item) => item.tool === tool.name)) {
        const chip = element('span', 'sc-action', pair.action);
        chip.style.setProperty('--dot', pair.color);
        actions.append(chip);
      }
      row.append(element('dt', '', tool.name), actions);
      list.append(row);
    }
    coverage.replaceChildren(element('p', 'sc-caption', 'Actions annotated for each instrument'), list);
  }

  function renderSources() {
    const counts = sourceVideos[split];
    const totalVideos = counts.reduce((sum, n) => sum + n, 0);
    const list = element('ul', 'sc-source-list');
    for (const [index, name] of sourceNames.entries()) {
      const count = counts[index];
      const row = element('li', 'sc-source-row');
      const track = element('span', 'sc-track');
      track.setAttribute('aria-hidden', 'true');
      const bar = element('span', 'sc-bar');
      bar.style.width = 100 * count / totalVideos + '%';
      track.append(bar);
      row.append(element('span', 'sc-source-name', name), track,
        element('span', 'sc-count', count + (count === 1 ? ' video' : ' videos')));
      list.append(row);
    }
    sources.replaceChildren(element('p', 'sc-caption', totalVideos + ' videos across five sources'), list);
  }

  function render() {
    cancelAnimationFrame(frame);
    const previouslyShowingDots = !testBody.hidden;
    selection = null;
    const showDots = split === 'test' && view !== 'source';
    title.textContent = 'Gold ' + splitNames[split] + ' set composition';
    testBody.hidden = !showDots;
    coverage.hidden = split === 'test' || view === 'source';
    sources.hidden = view !== 'source';
    for (const button of tabs) {
      const active = button.dataset.view === view;
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      if (active) panel.setAttribute('aria-labelledby', button.id);
    }
    if (view === 'source') renderSources();
    else if (!showDots) {
      if (view === 'pair') renderMatrix();
      else renderTools();
    } else {
      const previous = groups;
      groups = view === 'tool' ? toolGroups : pairGroups;
      const label = view === 'tool' ? 'Tool' : 'Tool–action pair';
      table.tHead.rows[0].cells[0].textContent = label;
      table.setAttribute('aria-label', 'Gold test clips by ' + label.toLowerCase());
      showLegend();
      resize();
      animate(340, previouslyShowingDots ? previous : null);
    }
  }

  function chooseView(next) {
    if (view === next) return;
    view = next;
    render();
  }

  tabs.forEach((button, index) => {
    button.addEventListener('click', () => chooseView(button.dataset.view));
    button.addEventListener('keydown', (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const targets = { ArrowLeft: (index + tabs.length - 1) % tabs.length,
        ArrowRight: (index + 1) % tabs.length, Home: 0, End: tabs.length - 1 };
      if (!Object.hasOwn(targets, event.key)) return;
      event.preventDefault();
      const next = tabs[targets[event.key]];
      next.focus({ preventScroll: true });
      chooseView(next.dataset.view);
    });
  });
  document.addEventListener('tactus:split-change', (event) => {
    const next = event.detail?.split;
    if (!Object.hasOwn(splitNames, next) || split === next) return;
    split = next;
    render();
  });
  function visitTestAnchor() {
    if (location.hash !== '#gold-test-set') return;
    view = 'pair';
    summary?.querySelector('[data-name="Test"]')?.click();
    split = 'test';
    render();
  }
  window.addEventListener('hashchange', visitTestAnchor);
  controls.hidden = false;
  figure.querySelector('.test-set__plot').hidden = !context;
  if (location.hash === '#gold-test-set') visitTestAnchor();
  else render();
  new ResizeObserver(resize).observe(canvas);
  reducedMotion.addEventListener('change', () => { cancelAnimationFrame(frame); draw(); });
})();

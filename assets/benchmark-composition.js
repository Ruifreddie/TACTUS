(() => {
  const figure = document.getElementById('gold-test-set');
  if (!figure) return;
  const summary = document.getElementById('benchmark-distribution');
  const table = figure.querySelector('#test-pair-counts');
  const body = table.tBodies[0];
  const controls = figure.querySelector('#composition-mode');
  const tabs = Array.from(controls.querySelectorAll('[data-view]'));
  const panel = figure.querySelector('#composition-panel');
  const coverage = figure.querySelector('#split-class-coverage');
  const sources = figure.querySelector('#split-sources');
  const splitNames = { train: 'Train', val: 'Val', test: 'Test' };
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
  const toolGroups = [];
  for (const pair of pairGroups) {
    const tool = toolGroups.find((group) => group.name === pair.tool);
    if (tool) tool.count += pair.count;
    else toolGroups.push({ name: pair.tool, count: pair.count, color: pair.color });
  }
  const total = pairGroups.reduce((sum, group) => sum + group.count, 0);
  function renderCounts() {
    const groups = view === 'tool' ? toolGroups : pairGroups;
    table.tHead.rows[0].cells[0].textContent = view === 'tool' ? 'Tool' : 'Tool–action pair';
    table.setAttribute('aria-label', view === 'tool' ? 'Gold test clips by tool' : 'Gold test clips by tool–action pair');
    body.replaceChildren();
    for (const group of groups) {
      const row = body.insertRow();
      row.insertCell().textContent = group.name;
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
    matrix.createCaption().textContent = splitNames[split] + ' · six annotated tool–action pairs';
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
    coverage.replaceChildren(element('p', 'sc-caption', splitNames[split] + ' · actions by instrument'), list);
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
    sources.replaceChildren(element('p', 'sc-caption', splitNames[split] + ' · ' + totalVideos + ' videos across five sources'), list);
  }

  function render() {
    table.hidden = split !== 'test' || view === 'source';
    coverage.hidden = split === 'test' || view === 'source';
    sources.hidden = view !== 'source';
    for (const button of tabs) {
      const active = button.dataset.view === view;
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      if (active) panel.setAttribute('aria-labelledby', button.id);
    }
    if (view === 'source') renderSources();
    else if (split === 'test') renderCounts();
    else if (view === 'pair') renderMatrix();
    else renderTools();
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
  if (location.hash === '#gold-test-set') visitTestAnchor();
  else render();
})();

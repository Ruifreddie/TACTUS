(() => {
  const summary = document.getElementById('benchmark-distribution');
  if (!summary) return;
  const controls = summary.querySelector('#composition-mode');
  const tabs = Array.from(controls.querySelectorAll('[data-view]'));
  const panel = summary.querySelector('#composition-panel');
  const table = panel.querySelector('#split-counts');
  const note = panel.querySelector('#split-count-note');
  const body = table.tBodies[0];
  const headers = table.tHead.rows[0].cells;
  const number = new Intl.NumberFormat('en-US');
  const splits = new Map(Array.from(summary.querySelectorAll('.bs-item'), (button) => [
    button.dataset.name.toLowerCase(),
    { name: button.dataset.name, clips: Number(button.dataset.clips), videos: Number(button.dataset.videos) }
  ]));
  const pairs = Array.from(body.rows, (row) => ({
    name: row.cells[0].textContent.trim(), tool: row.dataset.tool,
    testClips: Number(row.cells[1].textContent.replaceAll(',', ''))
  }));
  const tools = [...new Set(pairs.map((pair) => pair.tool))].map((name) => ({
    name,
    actions: pairs.filter((pair) => pair.tool === name).map((pair) => pair.name.split('–')[1]).join(' / '),
    testClips: pairs.filter((pair) => pair.tool === name).reduce((sum, pair) => sum + pair.testClips, 0)
  }));
  const sourceNames = ['Cholec80', 'SurgVU', 'YouTube', 'HeiChole', 'CRCD'];
  const sourceVideos = { train: [24, 26, 15, 8, 6], val: [3, 3, 3, 1, 1], test: [3, 3, 3, 1, 1] };
  const viewLabels = { pair: 'Tool–action pair', tool: 'Tool', source: 'Source' };
  let split = splits.has(summary.dataset.activeSplit) ? summary.dataset.activeSplit : 'train';
  let view = 'pair';

  function render() {
    const current = splits.get(split);
    const bySource = view === 'source';
    const total = bySource ? current.videos : current.clips;
    const rows = bySource
      ? sourceNames.map((name, i) => ({ name, count: sourceVideos[split][i] }))
      : (view === 'pair' ? pairs : tools).map((group) => ({
          name: group.name, actions: group.actions, count: split === 'test' ? group.testClips : null
        }));
    const unit = bySource ? 'videos' : 'clips';
    table.caption.textContent = current.name + ' · ' + number.format(total) + ' ' + unit;
    headers[0].textContent = viewLabels[view];
    headers[1].textContent = bySource ? 'Videos' : 'Clips';
    body.replaceChildren();
    for (const { name, actions, count } of rows) {
      const row = body.insertRow();
      const label = document.createElement('th');
      label.scope = 'row';
      label.textContent = name;
      if (actions) {
        const detail = document.createElement('span');
        detail.className = 'sc-tool-actions';
        detail.textContent = actions;
        label.append(detail);
      }
      row.append(label);
      row.insertCell().textContent = count === null ? '—' : number.format(count);
      row.insertCell().textContent = count === null ? '—' : (100 * count / total).toFixed(1) + '%';
    }
    note.hidden = bySource || split === 'test';
    if (note.hidden) table.removeAttribute('aria-describedby');
    else table.setAttribute('aria-describedby', note.id);
    for (const tab of tabs) {
      const active = tab.dataset.view === view;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active) panel.setAttribute('aria-labelledby', tab.id);
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
    if (!splits.has(next) || split === next) return;
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

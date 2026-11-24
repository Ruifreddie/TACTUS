(() => {
  const section = document.getElementById('dataset');
  if (!section) return;
  const tabs = Array.from(section.querySelectorAll('.ds-tab'));
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
  const charts = window.TactusPie;
  if (charts) {
    const pairRows = Array.from(section.querySelectorAll('.ds-pair-legend tbody tr'), row => {
      const label = row.cells[0].textContent.trim();
      const color = charts.color('pair', label);
      const swatch = row.querySelector('.pie-swatch');
      swatch.style.setProperty('--swatch', color);
      swatch.hidden = false;
      return { count: Number(row.cells[1].textContent.replaceAll(',', '')), color };
    });
    const pairPie = section.querySelector('.ds-pair-pie');
    charts.paint(pairPie, pairRows);
    pairPie.hidden = false;

    function paintShare(pie, share, color) {
      charts.paint(pie, [{ count: share, color }, { count: 100 - share, color: '#383a37' }]);
      pie.hidden = false;
    }

    section.querySelectorAll('.ds-visibility tbody tr').forEach(row => {
      const color = charts.color('tool', row.cells[0].textContent.trim());
      row.querySelectorAll('.ds-rate').forEach(cell => {
        const visible = Number.parseInt(cell.textContent, 10);
        paintShare(cell.querySelector('.ds-rate-pie'), visible, color);
      });
    });

    section.querySelectorAll('.ds-median-rate').forEach(rate => {
      const bin = section.querySelector('.ds-hist-plot .ds-' + rate.dataset.tier);
      paintShare(rate.querySelector('.ds-median-pie'), Number.parseInt(rate.querySelector('b').textContent, 10), getComputedStyle(bin).backgroundColor);
    });
  }
  function show(index) {
    tabs.forEach((tab, i) => {
      const selected = i === index;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panels[i].hidden = !selected;
    });
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => show(index));
    tab.addEventListener('keydown', event => {
      const targets = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 };
      if (!(event.key in targets)) return;
      event.preventDefault();
      const target = targets[event.key];
      show(target);
      tabs[target].focus();
    });
  });
  show(0);
})();

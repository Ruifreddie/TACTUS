(() => {
  const palette = {
    pair: {
      'hook-dissect': '#bd582d', 'grasper-grasp': '#427b71', 'grasper-dissect': '#6593aa',
      'scissors-dissect': '#827395', 'scissors-cut': '#c79b49', 'clipper-clip': '#b57179'
    },
    tool: { hook: '#bd582d', grasper: '#427b71', scissors: '#827395', clipper: '#b57179' },
    source: { cholec80: '#bd582d', surgvu: '#427b71', youtube: '#6593aa', heichole: '#827395', crcd: '#c79b49' }
  };

  function color(view, label) {
    const key = label.trim().toLowerCase().replace(/[\s–—]+/g, '-');
    return palette[view]?.[key] ?? '#88867d';
  }

  function paint(element, rows) {
    const slices = rows.filter(row => row.count > 0);
    const total = slices.reduce((sum, row) => sum + row.count, 0);
    if (!total) {
      element.style.removeProperty('--pie-fill');
      return;
    }
    let count = 0;
    const stops = slices.map(slice => {
      const start = count / total * 100;
      count += slice.count;
      return `${slice.color} ${start}% ${count / total * 100}%`;
    });
    element.style.setProperty('--pie-fill', `conic-gradient(${stops.join(', ')})`);
  }

  window.TactusPie = { color, paint };
})();

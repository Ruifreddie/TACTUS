(() => {
  const story = document.querySelector('.bs-story');
  if (!story) return;

  const card = story.querySelector('#bs-chart');
  const caption = card.querySelector('#bs-chart-caption');
  const buttons = [...story.querySelectorAll('.bs-item')];
  const number = new Intl.NumberFormat('en-US');
  const groups = buttons.map((button) => ({
    name: button.dataset.name,
    clips: Number(button.dataset.clips),
    videos: Number(button.dataset.videos),
    color: getComputedStyle(button).getPropertyValue('--bs-color').trim()
  }));
  const clipTotal = groups.reduce((sum, group) => sum + group.clips, 0);
  const videoTotal = groups.reduce((sum, group) => sum + group.videos, 0);
  const legendButtons = [...card.querySelectorAll('.bs-legend button')];
  const metrics = [...card.querySelectorAll('[data-metric]')].map((element) => ({
    key: element.dataset.metric,
    total: element.dataset.metric === 'clips' ? clipTotal : videoTotal,
    pie: element.querySelector('.bs-pie'),
    percentage: element.querySelector('.bs-pie-center b'),
    selectedName: element.querySelector('.bs-pie-split'),
    ratio: element.querySelector('.bs-ratio')
  }));

  if (window.TactusPie) {
    for (const metric of metrics) {
      window.TactusPie.paint(metric.pie, groups.map((group) => ({
        count: group[metric.key], color: group.color
      })));
    }
    card.classList.add('has-pies');
  }

  const percent = (count, total) => (100 * count / total).toFixed(2) + '%';
  let active = -1;
  function select(index) {
    if (active === index) return;
    active = index;
    const group = groups[index];
    buttons.forEach((button, i) => {
      button.classList.toggle('is-open', i === index);
      button.setAttribute('aria-expanded', String(i === index));
      legendButtons[i].parentElement.classList.toggle('is-current', i === index);
      legendButtons[i].setAttribute('aria-pressed', String(i === index));
    });
    for (const metric of metrics) {
      const count = group[metric.key];
      metric.percentage.textContent = percent(count, metric.total);
      metric.selectedName.textContent = group.name;
      metric.ratio.textContent = number.format(count) + ' of ' + number.format(metric.total);
    }
    caption.textContent = group.name + ' comprises ' + number.format(group.clips) + ' of '
      + number.format(clipTotal) + ' clips (' + percent(group.clips, clipTotal) + '), and '
      + group.videos + ' of ' + videoTotal + ' source videos (' + percent(group.videos, videoTotal) + ').';
    const split = group.name.toLowerCase();
    story.dataset.activeSplit = split;
    document.dispatchEvent(new CustomEvent('tactus:split-change', {
      detail: { split, name: group.name, clips: group.clips, videos: group.videos, clipTotal, videoTotal }
    }));
  }

  function bindSplitButton(button, index, controls) {
    button.addEventListener('click', () => select(index));
    button.addEventListener('keydown', (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      let next;
      switch (event.key) {
        case 'ArrowDown': case 'ArrowRight': next = (index + 1) % controls.length; break;
        case 'ArrowUp': case 'ArrowLeft': next = (index + controls.length - 1) % controls.length; break;
        case 'Home': next = 0; break;
        case 'End': next = controls.length - 1; break;
        default: return;
      }
      event.preventDefault();
      controls[next].focus({ preventScroll: true });
      select(next);
    });
  }
  buttons.forEach(bindSplitButton);
  legendButtons.forEach(bindSplitButton);
  select(0);
})();

(() => {
  const section = document.getElementById('dataset');
  if (!section) return;
  const tabs = Array.from(section.querySelectorAll('.ds-tab'));
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
  function show(index) {
    tabs.forEach((tab, i) => {
      const selected = i === index;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panels[i].hidden = !selected;
      panels[i].classList.toggle('ds-reveal', selected);
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
  const first = panels[0];
  first.classList.remove('ds-reveal');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        first.classList.add('ds-reveal');
        observer.disconnect();
      }
    }, { threshold: 0.1 });
    observer.observe(first);
  }
})();

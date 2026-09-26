'use strict';
// Resolve repository links from the current host, without storing an account or review ID.
const reviewPath = location.hostname === 'anonymous.4open.science'
  ? location.pathname.match(/^\/w\/([A-Za-z0-9_-]+)(?:\/|$)/) : null;
const projectName = location.pathname.split('/').filter(Boolean)[0];
let repositoryUrl;
if (reviewPath) {
  repositoryUrl = `${location.origin}/r/${reviewPath[1]}/`;
  const command = document.getElementById('quickstart');
  command.textContent = command.textContent.replace(
    '# Download this repository as TGPO.zip',
    `curl -fL ${location.origin}/api/repo/${reviewPath[1]}/zip -o TGPO.zip`
  );
} else if (location.hostname.endsWith('.github.io') && projectName) {
  const owner = location.hostname.slice(0, -'.github.io'.length);
  repositoryUrl = `https://github.com/${owner}/${encodeURIComponent(projectName)}/`;
}
if (repositoryUrl) {
  for (const link of document.querySelectorAll('[data-repository-path]')) {
    const path = link.dataset.repositoryPath;
    link.href = repositoryUrl + (reviewPath || !path ? '' : 'tree/main/') + path;
  }
}
for (const list of document.querySelectorAll('[role="tablist"]')) {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  function activate(tab, focus = false) {
    for (const item of tabs) {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
      document.getElementById(item.getAttribute('aria-controls')).hidden = !selected;
    }
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); activate(tabs[next], true); }
    });
  });
}
for (const button of document.querySelectorAll('[data-copy]')) {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(target.textContent);
      button.textContent = 'Copied';
      document.getElementById('copy-status').textContent = 'Copied to clipboard.';
      setTimeout(() => { button.textContent = 'Copy'; }, 1800);
    } catch {
      const range = document.createRange(); range.selectNodeContents(target);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      document.getElementById('copy-status').textContent = 'Text selected. Use your browser copy command.';
    }
  });
}

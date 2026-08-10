(function () {
  const SELECTOR = '.ai-message *';

  function tagDirection(root) {
    root.querySelectorAll(SELECTOR).forEach((el) => {
      if (el.getAttribute("dir") !== "auto") {
        el.setAttribute("dir", "auto");
      }
    });
  }

  // Debounce so rapid streaming-token mutations don't trigger excessive work
  let pending = null;
  function scheduleTag(root) {
    if (pending) return;
    pending = requestAnimationFrame(() => {
      tagDirection(root);
      pending = null;
    });
  }

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.addedNodes.length) {
        scheduleTag(document.body);
        return;
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Catch anything already rendered when this script first loads
  tagDirection(document.body);
})();
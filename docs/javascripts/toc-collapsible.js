/**
 * MkDocs Material: nested TOC entries (e.g. mkdocstrings class members) render as
 * always-visible sub-lists. Add per-section checkboxes like the rest of the nav
 * so top-level symbols stay closed until expanded.
 */
function enhanceMaterialToc() {
  document.querySelectorAll(".md-nav--secondary").forEach((toc) => {
    let gen = window.__mdTocCollapsibleSeq || 0;

    toc.querySelectorAll("li.md-nav__item").forEach((li) => {
      if (li.querySelector(":scope > input.md-nav__toggle")) {
        return;
      }
      const link = li.querySelector(":scope > a.md-nav__link");
      const sub = li.querySelector(":scope > nav.md-nav");
      if (!link || !sub) {
        return;
      }

      li.classList.add("md-nav__item--nested");
      const id = `__md_toc_col_${++gen}`;
      window.__mdTocCollapsibleSeq = gen;

      const input = document.createElement("input");
      input.type = "checkbox";
      input.className = "md-nav__toggle md-toggle";
      input.id = id;

      const label = document.createElement("label");
      label.className = "md-nav__link";
      label.setAttribute("for", id);
      label.innerHTML = link.innerHTML;
      const icon = document.createElement("span");
      icon.className = "md-nav__icon md-icon";
      label.appendChild(icon);

      li.insertBefore(input, link);
      li.insertBefore(label, link);
    });

    toc.querySelectorAll(".md-nav__link--active").forEach((a) => {
      let li = a.closest("li.md-nav__item");
      while (li && toc.contains(li)) {
        const t = li.querySelector(":scope > input.md-nav__toggle");
        if (t) {
          t.checked = true;
        }
        const parentList = li.parentElement;
        if (!parentList) {
          break;
        }
        li = parentList.closest("li.md-nav__item");
      }
    });
  });
}

(() => {
  // Material for MkDocs: re-run on instant navigation (no full page load)
  if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
    document$.subscribe(enhanceMaterialToc);
    return;
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhanceMaterialToc);
  } else {
    enhanceMaterialToc();
  }
})();

/* SPDX-License-Identifier: AGPL-3.0-only
 * Tiny hand-drawn SVG replacements for common UI icons. No web fonts required.
 * The remaining optional Font Awesome icons still work when its CDN loads. */
(() => {
    'use strict';
    const paths = {
        'fa-car-side': '<path d="M3 15V9l3-3h11l3 4 1 1v4H3z"/><path d="M7 6l-1 5h14"/><circle cx="7" cy="16" r="2"/><circle cx="18" cy="16" r="2"/>',
        'fa-car': '<path d="M4 15V9l3-3h10l3 3v6H4z"/><path d="M4 10h16"/><circle cx="7" cy="16" r="2"/><circle cx="17" cy="16" r="2"/>',
        'fa-chart-pie': '<path d="M11 2v10h10a10 10 0 1 1-10-10z"/><path d="M14 2a8 8 0 0 1 8 8h-8z"/>',
        'fa-video': '<rect x="2" y="6" width="14" height="12" rx="2"/><path d="m16 10 6-4v12l-6-4"/>',
        'fa-history': '<path d="M3 11a9 9 0 1 1 3 7"/><path d="M3 4v7h7"/><path d="M12 7v5l3 2"/>',
        'fa-search-plus': '<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6M10 7v6M7 10h6"/>',
        'fa-search': '<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
        'fa-road': '<path d="M8 3 3 21M16 3l5 18M12 3v4M12 11v3M12 18v3"/>',
        'fa-person-walking': '<circle cx="14" cy="4" r="2"/><path d="m12 9 3 3 4 1M12 9 8 13 4 13M12 9l-1 6-4 6M11 15l5 2 2 4"/>',
        'fa-images': '<rect x="5" y="5" width="16" height="16" rx="2"/><path d="M2 16V4a2 2 0 0 1 2-2h12M5 17l5-5 4 4 3-3 4 4"/><circle cx="15" cy="9" r="1"/>',
        'fa-image': '<rect x="2" y="3" width="20" height="18" rx="2"/><circle cx="16" cy="8" r="2"/><path d="m2 17 6-6 5 5 3-3 6 6"/>',
        'fa-flask': '<path d="M9 2h6M10 2v7L4 19a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3L14 9V2M7 15h10"/>',
        'fa-broadcast-tower': '<path d="M12 10v11M8 21h8M9 21l3-11 3 11"/><circle cx="12" cy="6" r="2"/><path d="M7 2a7 7 0 0 0 0 8M17 2a7 7 0 0 1 0 8M4 1a11 11 0 0 0 0 12M20 1a11 11 0 0 1 0 12"/>',
        'fa-cog': '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="8"/><path d="M12 1v3M12 20v3M1 12h3M20 12h3M4 4l2 2M18 18l2 2M20 4l-2 2M6 18l-2 2"/>',
        'fa-gear': '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="8"/><path d="M12 1v3M12 20v3M1 12h3M20 12h3"/>',
        'fa-scale-balanced': '<path d="M12 3v18M6 21h12M4 7h16M12 3v4M6 7l-4 8h8L6 7zM18 7l-4 8h8l-4-8z"/>',
        'fa-bars': '<path d="M4 6h16M4 12h16M4 18h16"/>',
        'fa-play': '<path d="m8 5 11 7-11 7z"/>',
        'fa-stop': '<rect x="5" y="5" width="14" height="14" rx="2"/>',
        'fa-bell': '<path d="M6 16V10a6 6 0 0 1 12 0v6l2 2H4zM10 21h4"/>',
        'fa-clock': '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
        'fa-calendar-day': '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 2v6M17 2v6M3 10h18M7 14h4M7 18h3"/>',
        'fa-chart-line': '<path d="M3 3v18h18M6 17l5-6 4 3 5-7"/>',
        'fa-chart-bar': '<path d="M3 3v18h18M7 18v-7M12 18V6M17 18v-10"/>',
        'fa-trash': '<path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14M10 11v6M14 11v6"/>',
        'fa-eye': '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
        'fa-check': '<path d="m4 12 5 5L20 6"/>',
        'fa-times': '<path d="M5 5l14 14M19 5 5 19"/>',
        'fa-xmark': '<path d="M5 5l14 14M19 5 5 19"/>',
        'fa-user': '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-5 3-8 8-8s8 3 8 8"/>',
        'fa-download': '<path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4"/>',
        'fa-upload': '<path d="M12 17V5m-5 5 5-5 5 5M4 17v4h16v-4"/>',
        'fa-info-circle': '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.5"/>',
        'fa-circle-info': '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.5"/>',
        'fa-triangle-exclamation': '<path d="M12 3 1 21h22L12 3zM12 9v5M12 18v.5"/>',
        'fa-exclamation-triangle': '<path d="M12 3 1 21h22L12 3zM12 9v5M12 18v.5"/>',
        'fa-filter': '<path d="M3 4h18l-7 8v7l-4 2v-9z"/>',
        'fa-spinner': '<path d="M12 3a9 9 0 1 1-9 9"/>',
    };
    const selector = 'i.fas, i.far, i.fa-solid';
    function renderIcons(root) {
        const elements = [];
        if (root.matches?.(selector)) elements.push(root);
        root.querySelectorAll?.(selector).forEach(el => elements.push(el));
        for (const element of elements) {
            const name = [...element.classList].find(key => Object.hasOwn(paths, key));
            if (!name) continue;
            element.innerHTML = `<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]}</svg>`;
            element.classList.remove('fas', 'far', 'fa-solid');
            element.classList.add('pv-icon');
            element.setAttribute('aria-hidden', 'true');
        }
    }
    document.addEventListener('DOMContentLoaded', () => {
        renderIcons(document);
        new MutationObserver(records => {
            for (const record of records) {
                for (const node of record.addedNodes) if (node.nodeType === 1) renderIcons(node);
            }
        }).observe(document.body, {childList: true, subtree: true});
    });
})();

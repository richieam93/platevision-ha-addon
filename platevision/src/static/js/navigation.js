/* SPDX-License-Identifier: AGPL-3.0-only
 * PlateVision 0.12.9: navigation-only enhancement. Never touches detection data.
 * Works offline and degrades to a fully expanded menu when JS is unavailable. */
(() => {
    'use strict';
    const FAVORITES_KEY = 'pv_nav_favorites_v1';
    const GROUPS_KEY = 'pv_nav_groups_v1';
    const MAX_FAVORITES = 6;

    function start() {
        const nav = document.getElementById('mainNavigation');
        const search = document.getElementById('sidebarNavSearch');
        if (!nav || !search || nav.dataset.enhanced === 'true') return;
        nav.dataset.enhanced = 'true';

        const entries = Array.from(nav.querySelectorAll('.pv-nav-entry[data-nav-page]'));
        const groups = Array.from(nav.querySelectorAll('.pv-nav-group[data-nav-group]'));
        const entryMap = new Map(entries.map(el => [el.dataset.navPage, el]));
        const favoritesBox = document.getElementById('pvNavFavorites');
        const favoritesList = document.getElementById('pvNavFavoriteLinks');
        const empty = document.getElementById('pvNavNoResults');
        const clear = document.getElementById('pvNavSearchClear');
        const hint = document.getElementById('pvNavMatchCount');
        const allButton = document.getElementById('pvNavExpandAll');
        const pagePin = document.getElementById('pvPageFavorite');
        const active = entries.find(el => el.querySelector('a[aria-current="page"]'));
        const activeGroup = active?.closest('.pv-nav-group');
        const tools = document.querySelector('.pv-nav-tools');
        const translations = {
            add: tools?.dataset.favoriteAdd || 'Als Favorit markieren',
            remove: tools?.dataset.favoriteRemove || 'Favorit entfernen',
            max: tools?.dataset.favoriteMax || 'Maximal sechs Favoriten möglich',
            results: tools?.dataset.resultsLabel || 'Treffer',
            showAll: tools?.dataset.showAll || 'Alle öffnen',
            hideAll: tools?.dataset.hideAll || 'Gruppen schließen',
        };

        function readStore(key, fallback) {
            try {
                const value = JSON.parse(window.localStorage.getItem(key) || 'null');
                return value !== null ? value : fallback;
            } catch (_) { return fallback; }
        }
        function writeStore(key, value) {
            try { window.localStorage.setItem(key, JSON.stringify(value)); }
            catch (_) { /* Storage disabled: current-page operation still works. */ }
        }
        const rawFavorites = readStore(FAVORITES_KEY, []);
        let favorites = Array.isArray(rawFavorites)
            ? [...new Set(rawFavorites.filter(id => typeof id === 'string' && entryMap.has(id)))].slice(0, MAX_FAVORITES)
            : [];
        const rawGroups = readStore(GROUPS_KEY, {});
        const expanded = (rawGroups && typeof rawGroups === 'object' && !Array.isArray(rawGroups)) ? rawGroups : {};
        const baseOpen = group => Object.hasOwn(expanded, group.dataset.navGroup)
            ? expanded[group.dataset.navGroup] === true
            : (group.dataset.navGroup === 'overview' || group === activeGroup);

        function setGroupOpen(group, open, persist = false) {
            const toggle = group.querySelector('.pv-nav-group-toggle');
            const items = group.querySelector('.pv-nav-group-items');
            if (!toggle || !items) return;
            items.hidden = !open;
            toggle.setAttribute('aria-expanded', String(open));
            if (persist) {
                expanded[group.dataset.navGroup] = open;
                writeStore(GROUPS_KEY, expanded);
            }
        }
        function restoreGroups(forceActive = false) {
            groups.forEach(group => setGroupOpen(group, (forceActive && group === activeGroup) || baseOpen(group)));
        }
        groups.forEach(group => {
            if (group === activeGroup) group.classList.add('pv-nav-current-group');
            group.querySelector('.pv-nav-group-toggle')?.addEventListener('click', () => {
                if (search.value.trim()) search.value = '';
                clear.hidden = true;
                groups.forEach(el => {
                    el.hidden = false;
                    el.querySelectorAll('.pv-nav-entry').forEach(item => { item.hidden = false; });
                });
                const currentlyOpen = group.querySelector('.pv-nav-group-toggle').getAttribute('aria-expanded') === 'true';
                setGroupOpen(group, !currentlyOpen, true);
                updateSearch();
            });
        });
        restoreGroups(true);

        function normalize(value) {
            return String(value || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
        }
        function updatePinButtons() {
            for (const entry of entries) {
                const button = entry.querySelector('.pv-nav-pin');
                const label = entry.querySelector('.pv-nav-entry-label')?.textContent?.trim() || '';
                if (!button) continue;
                const isFavorite = favorites.includes(entry.dataset.navPage);
                button.textContent = isFavorite ? '★' : '☆';
                button.setAttribute('aria-pressed', String(isFavorite));
                button.classList.toggle('pinned', isFavorite);
                const text = `${isFavorite ? translations.remove : translations.add}: ${label}`;
                button.setAttribute('aria-label', text);
                button.title = text;
            }
            if (pagePin) {
                const isFavorite = favorites.includes(pagePin.dataset.navPage);
                pagePin.textContent = isFavorite ? '★' : '☆';
                pagePin.classList.toggle('pinned', isFavorite);
                pagePin.setAttribute('aria-pressed', String(isFavorite));
                const pageLabel = entryMap.get(pagePin.dataset.navPage)?.querySelector('.pv-nav-entry-label')?.textContent?.trim() || '';
                const text = `${isFavorite ? translations.remove : translations.add}: ${pageLabel}`;
                pagePin.setAttribute('aria-label', text);
                pagePin.title = text;
            }
        }
        function renderFavorites() {
            if (!favoritesBox || !favoritesList) return;
            favoritesList.replaceChildren();
            favoritesBox.hidden = favorites.length === 0;
            for (const id of favorites) {
                const original = entryMap.get(id)?.querySelector('a.nav-link');
                const label = entryMap.get(id)?.querySelector('.pv-nav-entry-label')?.textContent?.trim();
                if (!original || !label) continue;
                const link = document.createElement('a');
                link.href = original.href;
                link.className = 'nav-link pv-nav-favorite-link' + (original.classList.contains('active') ? ' active' : '');
                link.dataset.favoritePage = id;
                if (original.hasAttribute('aria-current')) link.setAttribute('aria-current', 'page');
                const star = document.createElement('span');
                star.className = 'pv-favorite-mark';
                star.setAttribute('aria-hidden', 'true');
                star.textContent = '★';
                const title = document.createElement('span');
                title.textContent = label;
                title.className = 'pv-nav-entry-label';
                link.append(star, title);
                favoritesList.append(link);
            }
            updatePinButtons();
        }
        function toggleFavorite(id) {
            if (!entryMap.has(id)) return;
            if (favorites.includes(id)) favorites = favorites.filter(item => item !== id);
            else if (favorites.length < MAX_FAVORITES) favorites.push(id);
            else {
                if (typeof window.showToast === 'function') window.showToast('Navigation', translations.max, 'warning');
                return;
            }
            writeStore(FAVORITES_KEY, favorites);
            renderFavorites();
            updateSearch();
        }
        nav.addEventListener('click', event => {
            const pin = event.target.closest('button[data-nav-pin]');
            if (pin) { event.preventDefault(); toggleFavorite(pin.dataset.navPin); }
        });
        pagePin?.addEventListener('click', () => toggleFavorite(pagePin.dataset.navPage));
        renderFavorites();

        function updateSearch() {
            const query = normalize(search.value);
            let matches = 0;
            for (const group of groups) {
                let shown = 0;
                for (const entry of group.querySelectorAll('.pv-nav-entry')) {
                    const label = entry.querySelector('.pv-nav-entry-label')?.textContent || '';
                    const haystack = normalize(label + ' ' + (entry.dataset.navKeywords || ''));
                    const show = !query || haystack.includes(query);
                    entry.hidden = !show;
                    if (show) shown++;
                }
                group.hidden = shown === 0;
                if (query) setGroupOpen(group, shown > 0);
            }
            if (!query) restoreGroups();
            for (const link of favoritesList?.querySelectorAll('a') || []) {
                const show = !query || normalize(link.textContent).includes(query);
                link.hidden = !show;
            }
            if (favoritesBox) {
                favoritesBox.hidden = favorites.length === 0 || (!!query && !favoritesList?.querySelector('a:not([hidden])'));
            }
            matches = entries.filter(el => !el.hidden).length;
            if (empty) empty.hidden = !query || matches > 0;
            if (hint) hint.textContent = query ? `${matches} ${translations.results}` : (hint.dataset.initialLabel || '');
            if (clear) clear.hidden = !query;
            updateAllButton();
        }
        function updateAllButton() {
            if (!allButton) return;
            const everyOpen = groups.every(group => group.querySelector('.pv-nav-group-toggle')?.getAttribute('aria-expanded') === 'true');
            allButton.textContent = everyOpen ? translations.hideAll : translations.showAll;
        }
        function clearSearch() {
            search.value = '';
            updateSearch();
        }
        search.addEventListener('input', updateSearch);
        search.addEventListener('keydown', event => {
            if (event.key !== 'Enter' || !search.value.trim()) return;
            const visibleLinks = [...nav.querySelectorAll('.pv-nav-entry:not([hidden]) > a.nav-link')];
            if (visibleLinks.length === 1) { event.preventDefault(); visibleLinks[0].click(); }
        });
        clear?.addEventListener('click', () => { clearSearch(); search.focus(); });
        allButton?.addEventListener('click', () => {
            if (search.value) clearSearch();
            const shouldExpand = groups.some(group => group.querySelector('.pv-nav-group-toggle')?.getAttribute('aria-expanded') === 'false');
            groups.forEach(group => setGroupOpen(group, shouldExpand || group === activeGroup, true));
            updateAllButton();
        });
        document.addEventListener('keydown', event => {
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
                event.preventDefault();
                if (window.innerWidth < 992 && typeof window.setSidebarOpen === 'function') window.setSidebarOpen(true);
                search.focus();
                search.select();
            }
            if (event.key === 'Escape' && document.activeElement === search && search.value) {
                event.preventDefault();
                event.stopImmediatePropagation();
                clearSearch();
            }
        }, true);
        updateSearch();
        const alertBadge = document.getElementById('sidebarAlertCount');
        if (alertBadge && typeof window.pvUrl === 'function') {
            async function refreshAlertBadge() {
                if (document.hidden) return;
                try {
                    const response = await fetch(window.pvUrl('/api/alerts/status'), {cache:'no-store'});
                    if (!response.ok) throw new Error('Alert status unavailable');
                    const data = await response.json();
                    const count = Number(data.unread || 0);
                    alertBadge.textContent = count > 99 ? '99+' : String(Math.max(0, count));
                    alertBadge.style.display = count > 0 ? 'inline-block' : 'none';
                } catch (_) {
                    alertBadge.style.display = 'none';
                }
            }
            refreshAlertBadge();
            setInterval(refreshAlertBadge, 60000);
            document.addEventListener('visibilitychange', refreshAlertBadge);
        }
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
    else start();
})();

// PlateVision Pro+ shared helpers
window.PlateVision = {
  pct(value) { return `${Math.round(Number(value || 0) * 100)}%`; },
  formatDate(value) { return value ? new Date(value).toLocaleString('de-DE') : ''; },
  normalizePlate(text) { return String(text || '').toUpperCase().replace(/[^A-Z0-9ÄÖÜ]/g, ''); }
};

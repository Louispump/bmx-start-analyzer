/* ─────────────────────────────────────────────────────────────────
   Runtime i18n minimaliste — BMX Start Analyzer
   - Stockage du choix : localStorage `bmx-lang` ('fr' | 'en' | 'es')
   - Source de vérité dans le markup = français
   - data-i18n="key.path"          → remplace textContent
   - data-i18n-attr="attr:key,..." → remplace l'attribut (title, placeholder, …)
   - data-i18n-html="key"          → remplace innerHTML (à utiliser uniquement
     pour des strings que tu maîtrises, jamais user-generated)
   - window.t(key, fallback?)      → pour les strings JS dynamiques
   ───────────────────────────────────────────────────────────────── */
(function () {
  const STORAGE_KEY = 'bmx-lang';
  const DEFAULT_LANG = 'fr';
  const SUPPORTED = ['fr', 'en', 'es'];
  const LANG_NAMES = { fr: 'Français', en: 'English', es: 'Español' };

  let currentLang = DEFAULT_LANG;
  let dict = {};

  function detectLang() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && SUPPORTED.includes(stored)) return stored;
    } catch (e) {}
    return DEFAULT_LANG;
  }

  function resolve(key) {
    // key = "a.b.c" → dict.a.b.c
    if (!key) return null;
    const parts = key.split('.');
    let cur = dict;
    for (const p of parts) {
      if (cur == null) return null;
      cur = cur[p];
    }
    return (typeof cur === 'string') ? cur : null;
  }

  function tr(key, fallback) {
    const v = resolve(key);
    if (v != null) return v;
    if (fallback != null) return fallback;
    return key;
  }

  function applyAll() {
    // textContent
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const v   = resolve(key);
      if (v != null) el.textContent = v;
    });
    // innerHTML (réservé aux strings de confiance)
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      const key = el.getAttribute('data-i18n-html');
      const v   = resolve(key);
      if (v != null) el.innerHTML = v;
    });
    // Attributs : "title:key,placeholder:other"
    document.querySelectorAll('[data-i18n-attr]').forEach(el => {
      const spec = el.getAttribute('data-i18n-attr');
      spec.split(',').forEach(pair => {
        const [attr, key] = pair.split(':').map(s => s.trim());
        if (!attr || !key) return;
        const v = resolve(key);
        if (v != null) el.setAttribute(attr, v);
      });
    });
    // Met à jour <html lang="…"> pour aider la lecture vocale
    document.documentElement.setAttribute('lang', currentLang);
    // Émet un event pour que les scripts pages réagissent
    document.dispatchEvent(new CustomEvent('i18n:applied', { detail: { lang: currentLang } }));
  }

  async function load(lang) {
    if (lang === 'fr') {
      // FR = source de vérité dans le markup → pas de fetch
      dict = {};
      return;
    }
    try {
      const res = await fetch(`/static/i18n/${lang}.json`, { cache: 'no-cache' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      dict = await res.json();
    } catch (e) {
      console.warn('[i18n] Impossible de charger', lang, '— fallback FR.', e);
      dict = {};
    }
  }

  async function setLang(lang) {
    if (!SUPPORTED.includes(lang)) lang = DEFAULT_LANG;
    currentLang = lang;
    try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {}
    await load(lang);
    applyAll();
  }

  // API publique
  window.i18n = {
    get lang()    { return currentLang; },
    supported:    SUPPORTED,
    names:        LANG_NAMES,
    set:          setLang,
    apply:        applyAll,
  };
  window.t = tr;

  // Init dès que le DOM est dispo
  function init() {
    currentLang = detectLang();
    load(currentLang).then(applyAll);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

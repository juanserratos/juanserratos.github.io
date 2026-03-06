/*
 * render.js — Data-driven page renderer
 *
 * Each page includes this script with a data-page attribute:
 *   <script src="render.js" data-page="cv"></script>
 *
 * It fetches data/<page>.json, detects the language from <html lang>,
 * and renders the content into <div id="content">.
 */
(function () {
  var script = document.currentScript;
  var page = script.getAttribute('data-page');
  if (!page) return;

  var lang = document.documentElement.lang || 'en';
  // Determine base path to root (for pages in subdirectories like fr/, es/)
  var isSubdir = /^\/(fr|es)\//.test(location.pathname) ||
                 location.pathname.indexOf('/fr/') !== -1 ||
                 location.pathname.indexOf('/es/') !== -1;
  var base = isSubdir ? '../' : '';

  function t(obj) {
    if (!obj) return '';
    if (typeof obj === 'string') return obj;
    return obj[lang] || obj.en || '';
  }

  function esc(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  fetch(base + 'data/' + page + '.json')
    .then(function (r) { return r.json(); })
    .then(function (data) { render(data); })
    .catch(function (e) {
      var el = document.getElementById('content');
      if (el) el.innerHTML = '<p>Failed to load page data.</p>';
      console.error(e);
    });

  function render(data) {
    var el = document.getElementById('content');
    if (!el) return;
    var html = '';

    if (page === 'cv') html = renderCV(data);
    else if (page === 'publications') html = renderPublications(data);
    else if (page === 'books') html = renderBooks(data);
    else if (page === 'courses') html = renderCourses(data);
    else if (page === 'activities') html = renderActivities(data);

    el.innerHTML = html;

    // Trigger MathJax if present (publications page)
    if (window.MathJax && MathJax.Hub) {
      MathJax.Hub.Queue(['Typeset', MathJax.Hub, 'content']);
    }
  }

  /* ── CV ────────────────────────────────────────────────────────────────── */
  function renderCV(data) {
    var h = '<h1>' + t(data.title) + '</h1>';

    data.sections.forEach(function (sec) {
      h += '<h2>' + t(sec.heading) + '</h2>';

      if (sec.text) {
        h += '<p>' + t(sec.text) + '</p>';
        return;
      }

      if (sec.entries) {
        sec.entries.forEach(function (e) {
          h += '<div class="cv-entry">';
          h += '<div class="institution">' + t(e.institution);
          if (e.location) h += ' <span class="location">' + t(e.location) + '</span>';
          h += '</div>';
          if (e.role) h += '<div class="role">' + t(e.role) + '</div>';
          if (e.date) h += '<div class="date">' + t(e.date) + '</div>';
          if (e.bullets) {
            var bullets = t(e.bullets);
            if (Array.isArray(bullets) && bullets.length > 0) {
              h += '<ul>';
              bullets.forEach(function (b) { h += '<li>' + b + '</li>'; });
              h += '</ul>';
            }
          }
          h += '</div>';
        });
      }
    });

    if (data.cvDownload) {
      var pdfPath = base + 'cv.pdf';
      h += '<p style="margin-top: 40px;"><a href="' + pdfPath + '">' + t(data.cvDownload) + '</a></p>';
    }

    return h;
  }

  /* ── Publications ──────────────────────────────────────────────────────── */
  function renderPublications(data) {
    var h = '<h1>' + t(data.title) + '</h1>';

    data.sections.forEach(function (sec) {
      h += '<h2>' + t(sec.heading) + '</h2>';

      if (sec.articles) {
        sec.articles.forEach(function (a) {
          h += '<div class="pub-entry">';
          h += '<span class="pub-number">[' + a.number + ']</span> ';
          h += '<span class="pub-authors">' + esc(a.authors) + ':</span><br>';
          h += '<span class="pub-title">' + a.title + '</span><br>';
          h += '<span class="pub-venue">' + t(a.venue) + '</span>';
          if (a.link) {
            h += '<br><a href="' + a.link.url + '">&rsaquo; ' + esc(a.link.label) + '</a>';
          }
          h += '</div>';
        });
      }
    });

    if (data.researchAreas) {
      h += '<h2>' + t(data.researchAreas.heading) + '</h2>';
      h += '<p>' + t(data.researchAreas.text) + '</p>';
    }

    return h;
  }

  /* ── Books ─────────────────────────────────────────────────────────────── */
  function renderBooks(data) {
    var h = '<h1>' + t(data.title) + '</h1>';

    data.books.forEach(function (b) {
      var tagClass = b.status === 'reading' ? 'tag-reading' :
                     b.status === 'read'    ? 'tag-read' :
                                              'tag-want';
      h += '<div class="book-entry">';
      h += '<div class="book-header" onclick="this.parentElement.classList.toggle(\'open\')">';
      h += '<span class="book-toggle">+</span>';
      h += '<span class="book-title">' + esc(b.title) + '</span>';
      h += '<span class="book-author">' + esc(b.author) + '</span>';
      h += '<span class="book-tag ' + tagClass + '">' + t(b.statusLabel) + '</span>';
      h += '</div>';
      h += '<div class="book-notes"><p>' + (b.notes || '') + '</p></div>';
      h += '</div>';
    });

    return h;
  }

  /* ── Courses ───────────────────────────────────────────────────────────── */
  function renderCourses(data) {
    var h = '<h1>' + t(data.title) + '</h1>';

    data.sections.forEach(function (sec) {
      h += '<h2>' + t(sec.heading) + '</h2>';
      if (sec.subtitle) {
        h += '<p style="color: var(--muted); margin-bottom: 20px;">' + t(sec.subtitle) + '</p>';
      }
      if (sec.note) {
        h += '<p>' + t(sec.note) + '</p>';
      }
      if (sec.courses) {
        sec.courses.forEach(function (c) {
          h += '<div class="course-entry">';
          h += '<div class="course-title">' + t(c.title) + '</div>';
          h += '<div class="course-meta">' + t(c.meta) + '</div>';
          h += '</div>';
        });
      }
    });

    return h;
  }

  /* ── Activities ────────────────────────────────────────────────────────── */
  function renderActivities(data) {
    var h = '<h1>' + t(data.title) + '</h1>';

    data.sections.forEach(function (sec) {
      h += '<h2>' + t(sec.heading) + '</h2>';

      sec.entries.forEach(function (e) {
        h += '<div class="activity-entry">';
        h += '<div class="activity-title">' + t(e.title) + '</div>';
        h += '<div class="activity-meta">' + t(e.meta) + '</div>';
        if (e.description) {
          h += '<p style="margin-top: 4px; font-size: 0.95em;">';
          h += t(e.description);
          if (e.link) {
            h += ' <a href="' + e.link.url + '">&rsaquo; ' + t(e.link.label) + '</a>';
          }
          h += '</p>';
        }
        h += '</div>';
      });
    });

    return h;
  }
})();

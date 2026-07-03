/**
 * Markdown rendering for assistant messages — escape-first, zero-dependency.
 *
 * The model's raw text is HTML-escaped BEFORE any markup is generated, so the
 * only tags that ever reach innerHTML are the ones this module builds. A
 * crafted response cannot inject HTML or script.
 *
 * Supported: fenced code blocks (language label + copy button), inline code,
 * headings, bold / italic / strikethrough, http(s) links, ordered and
 * unordered lists, blockquotes, tables, horizontal rules. An unclosed fence
 * renders as an open code block, so streaming output looks right mid-fence.
 */

const ESCAPES: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
};

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

/** Inline markup on an already-escaped string. Code spans are carved out
 *  first so bold/italic/link rules never fire inside them. */
function renderInline(escaped: string): string {
  const parts = escaped.split(/(`+[^`]*`+)/g);
  return parts.map((part, i) => {
    if (i % 2 === 1) {
      const inner = part.replace(/^`+/, '').replace(/`+$/, '');
      return `<code>${inner}</code>`;
    }
    return part
      // links before emphasis: label may contain * that isn't emphasis
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>')
      .replace(/~~([^~]+)~~/g, '<del>$1</del>');
  }).join('');
}

const RE_FENCE_OPEN = /^\s*```+\s*([\w+#.-]*)\s*$/;
const RE_FENCE_CLOSE = /^\s*```+\s*$/;
const RE_HEADING = /^(#{1,6})\s+(.*)$/;
const RE_HR = /^ {0,3}([-*_]) *(?:\1 *){2,}$/;
const RE_UL = /^\s{0,8}[-*+]\s+(.*)$/;
const RE_OL = /^\s{0,8}\d{1,3}[.)]\s+(.*)$/;
const RE_QUOTE = /^\s{0,3}>\s?(.*)$/;
const RE_TABLE_SEP = /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/;

function isTableRow(line: string): boolean {
  return line.includes('|') && line.trim().length > 0;
}

function splitTableRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|')) s = s.slice(0, -1);
  return s.split('|').map(c => c.trim());
}

/** Convert markdown source to safe HTML. */
export function markdownToHtml(src: string): string {
  const lines = src.split('\n');
  const out: string[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block — an unclosed fence runs to EOF (streaming-friendly)
    const fence = line.match(RE_FENCE_OPEN);
    if (fence) {
      const lang = fence[1];
      const code: string[] = [];
      i++;
      while (i < lines.length && !RE_FENCE_CLOSE.test(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      i++; // past the closing fence (or EOF)
      out.push(
        `<div class="md-code"><div class="md-code-head"><span>${escapeHtml(lang || 'code')}</span>`
        + `<button class="md-copy" type="button">copy</button></div>`
        + `<pre>${escapeHtml(code.join('\n'))}</pre></div>`);
      continue;
    }

    if (line.trim() === '') { i++; continue; }

    const heading = line.match(RE_HEADING);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}>${renderInline(escapeHtml(heading[2]))}</h${level}>`);
      i++;
      continue;
    }

    if (RE_HR.test(line)) { out.push('<hr>'); i++; continue; }

    const quoteM = line.match(RE_QUOTE);
    if (quoteM) {
      const quoted: string[] = [];
      while (i < lines.length) {
        const m = lines[i].match(RE_QUOTE);
        if (!m) break;
        quoted.push(renderInline(escapeHtml(m[1])));
        i++;
      }
      out.push(`<blockquote>${quoted.join('<br>')}</blockquote>`);
      continue;
    }

    if (RE_UL.test(line) || RE_OL.test(line)) {
      const ordered = RE_OL.test(line);
      const re = ordered ? RE_OL : RE_UL;
      const items: string[] = [];
      while (i < lines.length) {
        const m = lines[i].match(re);
        if (!m) break;
        items.push(`<li>${renderInline(escapeHtml(m[1]))}</li>`);
        i++;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.join('')}</${tag}>`);
      continue;
    }

    // Table: a header row followed by a |---|---| separator
    if (isTableRow(line) && i + 1 < lines.length
        && RE_TABLE_SEP.test(lines[i + 1]) && lines[i + 1].includes('|')) {
      const header = splitTableRow(line)
        .map(c => `<th>${renderInline(escapeHtml(c))}</th>`).join('');
      i += 2;
      const rows: string[] = [];
      while (i < lines.length && isTableRow(lines[i]) && !RE_HEADING.test(lines[i])) {
        const cells = splitTableRow(lines[i])
          .map(c => `<td>${renderInline(escapeHtml(c))}</td>`).join('');
        rows.push(`<tr>${cells}</tr>`);
        i++;
      }
      out.push(
        `<div class="md-table-wrap"><table><thead><tr>${header}</tr></thead>`
        + `<tbody>${rows.join('')}</tbody></table></div>`);
      continue;
    }

    // Paragraph: consecutive plain lines, hard-broken with <br>
    const para: string[] = [];
    while (i < lines.length) {
      const l = lines[i];
      if (l.trim() === '' || RE_FENCE_OPEN.test(l) || RE_HEADING.test(l)
          || RE_HR.test(l) || RE_QUOTE.test(l) || RE_UL.test(l) || RE_OL.test(l)) break;
      para.push(renderInline(escapeHtml(l)));
      i++;
    }
    out.push(`<p>${para.join('<br>')}</p>`);
  }

  return out.join('');
}

/** Render markdown into an element (replaces its content). */
export function renderMarkdown(el: HTMLElement, src: string): void {
  el.innerHTML = markdownToHtml(src);
}

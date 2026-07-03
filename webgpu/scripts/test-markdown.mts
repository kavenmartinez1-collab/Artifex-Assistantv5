/**
 * Unit tests for src/chat/markdown.ts — the escape-first renderer behind
 * assistant messages. The escape cases are the load-bearing ones: model
 * output must never be able to inject markup. Run:
 *     npx tsx scripts/test-markdown.mts
 */
import { escapeHtml, markdownToHtml } from '../src/chat/markdown';

let pass = 0;
let fail = 0;

function check(label: string, actual: string, expected: string) {
  if (actual === expected) {
    pass++;
  } else {
    fail++;
    console.error(`FAIL ${label}`);
    console.error(`  expected: ${expected}`);
    console.error(`  actual:   ${actual}`);
  }
}

function checkContains(label: string, actual: string, needle: string) {
  if (actual.includes(needle)) {
    pass++;
  } else {
    fail++;
    console.error(`FAIL ${label}`);
    console.error(`  missing:  ${needle}`);
    console.error(`  actual:   ${actual}`);
  }
}

function checkNotContains(label: string, actual: string, needle: string) {
  if (!actual.includes(needle)) {
    pass++;
  } else {
    fail++;
    console.error(`FAIL ${label}: must not contain ${needle}`);
    console.error(`  actual:   ${actual}`);
  }
}

// ── Escaping: raw HTML in model output must never survive ──────────────────
check('escapeHtml', escapeHtml(`<img src=x onerror="a('b')">&`),
  '&lt;img src=x onerror=&quot;a(&#39;b&#39;)&quot;&gt;&amp;');
checkNotContains('script tag escaped', markdownToHtml('<script>alert(1)</script>'), '<script');
checkNotContains('img tag escaped', markdownToHtml('hi <img src=x onerror=alert(1)>'), '<img');
checkNotContains('tag inside code block escaped', markdownToHtml('```\n<script>x</script>\n```'), '<script');
checkNotContains('tag inside inline code escaped', markdownToHtml('run `<b>bold</b>` now'), '<b>');
checkNotContains('javascript: link rejected', markdownToHtml('[x](javascript:alert(1))'), 'href');
checkNotContains('attr breakout via link label', markdownToHtml('["><script>x</script>](https://a.b)'), '<script');

// ── Block elements ──────────────────────────────────────────────────────────
check('paragraph', markdownToHtml('hello world'), '<p>hello world</p>');
check('hard break', markdownToHtml('a\nb'), '<p>a<br>b</p>');
check('two paragraphs', markdownToHtml('a\n\nb'), '<p>a</p><p>b</p>');
check('heading', markdownToHtml('## Title'), '<h2>Title</h2>');
check('hr', markdownToHtml('---'), '<hr>');
check('ul', markdownToHtml('- a\n- b'), '<ul><li>a</li><li>b</li></ul>');
check('ol', markdownToHtml('1. a\n2. b'), '<ol><li>a</li><li>b</li></ol>');
check('blockquote', markdownToHtml('> a\n> b'), '<blockquote>a<br>b</blockquote>');
checkContains('code block with lang', markdownToHtml('```python\nprint(1)\n```'),
  '<span>python</span>');
checkContains('code block content', markdownToHtml('```python\nprint(1)\n```'),
  '<pre>print(1)</pre>');
checkContains('code block copy button', markdownToHtml('```\nx\n```'), 'md-copy');
checkContains('unclosed fence still renders (streaming)', markdownToHtml('```js\nlet x = 1'),
  '<pre>let x = 1</pre>');
checkContains('table', markdownToHtml('| a | b |\n|---|---|\n| 1 | 2 |'),
  '<thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody>');

// ── Inline elements ─────────────────────────────────────────────────────────
check('bold', markdownToHtml('**hi**'), '<p><strong>hi</strong></p>');
check('italic', markdownToHtml('so *nice* yes'), '<p>so <em>nice</em> yes</p>');
check('strike', markdownToHtml('~~gone~~'), '<p><del>gone</del></p>');
check('inline code', markdownToHtml('use `x = 1` here'), '<p>use <code>x = 1</code> here</p>');
check('link', markdownToHtml('[hi](https://a.b/c)'),
  '<p><a href="https://a.b/c" target="_blank" rel="noopener noreferrer">hi</a></p>');
check('no emphasis inside code', markdownToHtml('`a * b * c`'), '<p><code>a * b * c</code></p>');
check('multiplication is not italic', markdownToHtml('2 * 3 * 4'), '<p>2 * 3 * 4</p>');

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail > 0 ? 1 : 0);

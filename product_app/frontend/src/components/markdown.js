import MarkdownIt from "markdown-it";
import DOMPurify from "dompurify";
import katex from "katex";

const md = new MarkdownIt({
  breaks: true,
  html: false,
  linkify: true,
});

// Counseling UI does not need strikethrough — disable GFM-style del.
md.disable("strikethrough");

function escapeLoneTildes(text) {
  const s = String(text || "");
  let out = "";
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "~" && s[i - 1] !== "~" && s[i + 1] !== "~") out += "\\~";
    else out += s[i];
  }
  return out;
}

function renderKatexInHtml(html) {
  let out = html;
  out = out.replace(/\$\$([\s\S]+?)\$\$/g, (full, expr) => {
    try {
      return katex.renderToString(expr, { displayMode: true, throwOnError: false });
    } catch {
      return full;
    }
  });
  out = out.replace(/\$([^$\n]+?)\$/g, (full, expr) => {
    try {
      return katex.renderToString(expr, { displayMode: false, throwOnError: false });
    } catch {
      return full;
    }
  });
  return out;
}

export function renderAssistantMarkdown(text) {
  const raw = escapeLoneTildes(String(text || ""));
  const html = DOMPurify.sanitize(md.render(raw));
  return renderKatexInHtml(html);
}

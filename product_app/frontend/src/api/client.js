export async function api(url, options = {}) {
  const opts = {
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  };
  const r = await fetch(url, opts);
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return r.json();
  }
  return { ok: r.ok, status: r.status, text: await r.text() };
}

export function postJson(url, body) {
  return api(url, { method: "POST", body: JSON.stringify(body ?? {}) });
}

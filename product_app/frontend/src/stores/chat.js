import { defineStore } from "pinia";
import * as chatApi from "@/api/chat";

function historyLooksPending(messages) {
  if (!messages || messages.length === 0) return false;
  const last = messages[messages.length - 1];
  if (last.role === "user") return true;
  if (last.role === "assistant" && !String(last.content || "").trim()) return true;
  return false;
}

export const useChatStore = defineStore("chat", {
  state: () => ({
    username: "",
    conversations: [],
    messages: [],
    isStreaming: false,
    streamAbort: null,
    pollTimer: null,
    sendingLabel: "发送",
  }),
  actions: {
    async loadMe() {
      const j = await chatApi.getMe();
      if (j.ok) this.username = j.username || "";
      return j;
    },
    abortStream() {
      if (this.streamAbort) {
        this.streamAbort.abort();
        this.streamAbort = null;
      }
    },
    stopHistoryPoll() {
      if (this.pollTimer) {
        clearInterval(this.pollTimer);
        this.pollTimer = null;
      }
    },
    scheduleHistoryPoll() {
      this.stopHistoryPoll();
      let attempts = 0;
      this.pollTimer = setInterval(async () => {
        attempts += 1;
        const r = await chatApi.getHistory();
        if (!r.ok) return;
        if (!historyLooksPending(r.messages)) {
          this.stopHistoryPoll();
          await this.loadHistory(false);
          await this.loadConversations();
          return;
        }
        if (attempts >= 90) this.stopHistoryPoll();
      }, 2000);
    },
    async loadHistory(startPoll = true) {
      const j = await chatApi.getHistory();
      if (!j.ok) return;
      this.messages = (j.messages || []).map((m) => ({
        role: m.role,
        content: m.content || "",
        pending: m.role === "assistant" && !String(m.content || "").trim(),
      }));
      if (startPoll && historyLooksPending(j.messages)) this.scheduleHistoryPoll();
      else if (!historyLooksPending(j.messages)) this.stopHistoryPoll();
    },
    async loadConversations() {
      const j = await chatApi.getConversations();
      if (!j.ok) return;
      this.conversations = j.conversations || [];
    },
    async selectConversation(sid) {
      this.abortStream();
      this.stopHistoryPoll();
      await chatApi.selectConversation(sid);
      await this.loadHistory();
      await this.loadConversations();
    },
    async newConversation() {
      this.abortStream();
      this.stopHistoryPoll();
      await chatApi.newConversation();
      await this.loadHistory();
      await this.loadConversations();
    },
    async renameConversation(sid, title) {
      await chatApi.renameConversation(sid, title);
      await this.loadConversations();
    },
    async deleteConversation(sid) {
      const j = await chatApi.deleteConversation(sid);
      if (j.rotated) await this.loadHistory();
      await this.loadConversations();
    },
    async logout() {
      this.abortStream();
      this.stopHistoryPoll();
      await chatApi.logout();
    },
    async sendMessage(text) {
      const t = String(text || "").trim();
      if (!t || this.isStreaming) return;

      this.abortStream();
      this.stopHistoryPoll();
      this.messages.push({ role: "user", content: t });
      this.isStreaming = true;
      this.sendingLabel = "思考中…";

      const controller = new AbortController();
      this.streamAbort = controller;
      // Must mutate via the reactive array entry (Proxy). Mutating a raw object
      // that was pushed in does not trigger Vue updates, so the UI would only
      // refresh after the stream finishes — looking like a non-streaming blob.
      this.messages.push({
        role: "assistant",
        content: "",
        pending: true,
        streaming: true,
      });
      const placeholder = this.messages[this.messages.length - 1];

      try {
        const r = await fetch("/api/chat_stream", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: t }),
          signal: controller.signal,
        });

        if (!r.ok || !r.body) {
          const j2 = await chatApi.chatOnce(t);
          placeholder.content = j2.ok
            ? j2.assistant || ""
            : `请求失败：${j2.error || "unknown"}`;
          placeholder.pending = false;
          placeholder.streaming = false;
          await this.loadConversations();
          return;
        }

        const reader = r.body.getReader();
        const decoder = new TextDecoder("utf-8");
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          if (!chunk) continue;
          placeholder.content += chunk;
          placeholder.pending = false;
        }
        if (!String(placeholder.content || "").trim()) {
          placeholder.pending = true;
        }
        placeholder.streaming = false;
        await this.loadConversations();
      } catch (e) {
        if (e && e.name === "AbortError") return;
        this.messages.push({ role: "assistant", content: `请求异常：${String(e)}` });
      } finally {
        if (this.streamAbort === controller) this.streamAbort = null;
        this.isStreaming = false;
        this.sendingLabel = "发送";
      }
    },
  },
});

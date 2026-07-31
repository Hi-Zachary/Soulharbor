import { defineStore } from "pinia";
import * as memApi from "@/api/memory";

export const usePrivacyStore = defineStore("privacy", {
  state: () => ({
    enabled: false,
    busy: false,
    modalOpen: false,
    toast: "",
    forgetConfirm: false,
    hint: "加载中…",
  }),
  actions: {
    showToast(msg) {
      this.toast = msg;
      clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => {
        this.toast = "";
      }, 2200);
    },
    applyEnabled(enabled) {
      this.enabled = !!enabled;
      this.hint = enabled
        ? "已开启：对话中的有用线索可能被记住，用于后续支持"
        : "已关闭：不会再写入或召回长期记忆";
    },
    open() {
      this.forgetConfirm = false;
      this.toast = "";
      this.modalOpen = true;
    },
    close() {
      this.modalOpen = false;
      this.forgetConfirm = false;
    },
    async load() {
      const s = await memApi.getMemorySettings();
      if (!s || !s.ok) {
        this.hint = "无法加载设置";
        return false;
      }
      this.applyEnabled(!!s.memory_enabled);
      return true;
    },
    async toggle() {
      if (this.busy) return;
      this.busy = true;
      try {
        const j = this.enabled ? await memApi.disableMemory() : await memApi.enableMemory();
        if (!j.ok) {
          this.showToast("操作失败，请稍后重试");
          return;
        }
        this.applyEnabled(!this.enabled);
        this.showToast(this.enabled ? "已开启长期记忆" : "已关闭长期记忆");
      } catch (_) {
        this.showToast("网络异常，请稍后重试");
      } finally {
        this.busy = false;
      }
    },
    async forgetAll() {
      if (this.busy) return;
      this.busy = true;
      try {
        const j = await memApi.forgetAllMemory();
        this.forgetConfirm = false;
        this.showToast(j.ok ? "已清除全部长期记忆" : "清除失败");
      } catch (_) {
        this.showToast("网络异常，请稍后重试");
      } finally {
        this.busy = false;
      }
    },
  },
});

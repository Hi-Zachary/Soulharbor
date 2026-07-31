import { defineStore } from "pinia";
import * as chatApi from "@/api/chat";

export const useAuthStore = defineStore("auth", {
  state: () => ({
    username: "",
    ready: false,
  }),
  actions: {
    async refresh() {
      try {
        const j = await chatApi.getMe();
        if (j.ok) {
          this.username = j.username || "";
          this.ready = true;
          return true;
        }
      } catch (_) {
        /* ignore */
      }
      this.username = "";
      this.ready = true;
      return false;
    },
    async login(username, password) {
      return chatApi.login(username, password);
    },
    async register(username, password) {
      return chatApi.register(username, password);
    },
    async logout() {
      await chatApi.logout();
      this.username = "";
    },
  },
});

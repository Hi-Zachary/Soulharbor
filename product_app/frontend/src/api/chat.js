import { api, postJson } from "./client";

export const getMe = () => api("/api/me");
export const login = (username, password) => postJson("/api/login", { username, password });
export const register = (username, password) => postJson("/api/register", { username, password });
export const logout = () => postJson("/api/logout", {});
export const getHistory = () => api("/api/history");
export const getConversations = () => api("/api/conversations");
export const newConversation = () => postJson("/api/conversation/new", {});
export const selectConversation = (sid) => postJson("/api/conversation/select", { sid });
export const renameConversation = (sid, title) =>
  postJson("/api/conversation/rename", { sid, title });
export const deleteConversation = (sid) => postJson("/api/conversation/delete", { sid });
export const chatOnce = (text) => postJson("/api/chat", { text });

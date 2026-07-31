import { api, postJson } from "./client";

export const getMemorySettings = () => api("/api/memory/settings");
export const enableMemory = () => postJson("/api/memory/enable", {});
export const disableMemory = () => postJson("/api/memory/disable", {});
export const forgetAllMemory = () => postJson("/api/memory/forget-all", {});

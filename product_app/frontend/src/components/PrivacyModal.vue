<script setup>
import { onMounted, onUnmounted } from "vue";
import { usePrivacyStore } from "@/stores/privacy";

const privacy = usePrivacyStore();

function onKey(e) {
  if (e.key === "Escape" && privacy.modalOpen) privacy.close();
}

onMounted(() => document.addEventListener("keydown", onKey));
onUnmounted(() => document.removeEventListener("keydown", onKey));
</script>

<template>
  <div
    v-if="privacy.modalOpen"
    class="fixed inset-0 z-50"
    aria-hidden="false"
  >
    <div class="absolute inset-0 bg-slate-900/40 backdrop-blur-[2px]" @click="privacy.close()"></div>
    <div class="relative flex min-h-full items-center justify-center p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="privacy-title"
        class="w-full max-w-md rounded-2xl border border-teal-100 bg-white shadow-xl shadow-slate-900/10"
      >
        <div class="flex items-start justify-between gap-3 border-b border-slate-100 px-5 py-4">
          <div>
            <h2 id="privacy-title" class="text-base font-bold text-slate-800">隐私与记忆</h2>
            <p class="mt-1 text-xs leading-5 text-slate-500">
              用于跨会话个性化支持；具体记忆内容不会在聊天页展示。
            </p>
          </div>
          <button
            type="button"
            class="rounded-lg px-2 py-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition-colors"
            aria-label="关闭"
            @click="privacy.close()"
          >
            ✕
          </button>
        </div>

        <div class="space-y-4 px-5 py-5">
          <div
            class="flex items-center justify-between gap-4 rounded-xl border border-slate-100 bg-slate-50/80 px-4 py-3.5"
          >
            <div class="min-w-0">
              <div class="text-sm font-semibold text-slate-800">长期记忆</div>
              <div class="mt-0.5 text-xs text-slate-500">{{ privacy.hint }}</div>
            </div>
            <button
              type="button"
              role="switch"
              :aria-checked="privacy.enabled ? 'true' : 'false'"
              class="relative h-7 w-12 shrink-0 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-teal-500/30"
              :class="privacy.enabled ? 'bg-teal-500' : 'bg-slate-300'"
              :disabled="privacy.busy"
              @click="privacy.toggle()"
            >
              <span
                class="absolute left-0.5 top-0.5 h-6 w-6 rounded-full bg-white shadow transition-transform"
                :style="{ transform: privacy.enabled ? 'translateX(1.25rem)' : 'translateX(0)' }"
              ></span>
            </button>
          </div>

          <div class="rounded-xl border border-rose-100 bg-rose-50/50 px-4 py-3.5">
            <div class="text-sm font-semibold text-slate-800">清除全部长期记忆</div>
            <p class="mt-1 text-xs leading-5 text-slate-500">
              删除后不可恢复。关闭长期记忆不会自动清空已有内容。
            </p>
            <button
              type="button"
              class="mt-3 w-full rounded-xl border border-rose-200 bg-white px-4 py-2.5 text-sm font-medium text-rose-600 hover:bg-rose-50 transition-colors"
              @click="privacy.forgetConfirm = !privacy.forgetConfirm"
            >
              清除全部记忆
            </button>
            <div v-if="privacy.forgetConfirm" class="mt-3 space-y-2">
              <p class="text-xs text-rose-700">确定清除？此操作不可撤销。</p>
              <div class="flex gap-2">
                <button
                  type="button"
                  class="flex-1 rounded-xl bg-rose-500 px-3 py-2 text-sm font-medium text-white hover:bg-rose-600 transition-colors"
                  @click="privacy.forgetAll()"
                >
                  确认清除
                </button>
                <button
                  type="button"
                  class="flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 transition-colors"
                  @click="privacy.forgetConfirm = false"
                >
                  取消
                </button>
              </div>
            </div>
          </div>

          <p v-if="privacy.toast" class="text-center text-xs font-medium text-teal-700">
            {{ privacy.toast }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

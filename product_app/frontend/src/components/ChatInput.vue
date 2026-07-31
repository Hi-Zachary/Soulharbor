<script setup>
import { ref } from "vue";

const emit = defineEmits(["send"]);
defineProps({
  disabled: { type: Boolean, default: false },
  sendingLabel: { type: String, default: "发送" },
});

const text = ref("");

function submit() {
  const t = text.value.trim();
  if (!t) return;
  emit("send", t);
  text.value = "";
}

function onKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submit();
  }
}
</script>

<template>
  <div class="p-4 border-t border-gray-100 bg-white">
    <div class="flex gap-3">
      <input
        v-model="text"
        class="flex-1 rounded-xl bg-gray-50 border border-gray-200 px-4 py-3 outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 transition-all text-slate-700"
        placeholder="分享你的心情，或者任何想说的话…"
        :disabled="disabled"
        @keydown="onKeydown"
      />
      <button
        class="px-6 py-3 rounded-xl bg-teal-500 hover:bg-teal-600 text-white font-medium shadow-md shadow-teal-500/20 transition-all whitespace-nowrap"
        :class="{ 'opacity-70': disabled }"
        :disabled="disabled"
        @click="submit"
      >
        {{ sendingLabel }}
      </button>
    </div>
    <div class="mt-3 text-xs text-slate-400 text-center">
      免责声明：本产品用于心理支持与信息服务，不替代专业医疗诊断；如处于紧急危险请联系当地紧急服务或身边可信任的人。
    </div>
  </div>
</template>

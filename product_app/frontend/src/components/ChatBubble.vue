<script setup>
import { computed } from "vue";
import { renderAssistantMarkdown } from "./markdown";
import ThinkingIndicator from "./ThinkingIndicator.vue";

const props = defineProps({
  role: { type: String, required: true },
  content: { type: String, default: "" },
  pending: { type: Boolean, default: false },
});

const html = computed(() =>
  props.role === "assistant" && !props.pending ? renderAssistantMarkdown(props.content) : ""
);
</script>

<template>
  <div class="flex w-full" :class="role === 'user' ? 'justify-end' : 'justify-start'">
    <div
      class="max-w-[85%] rounded-2xl px-5 py-3 leading-relaxed shadow-sm text-sm md:text-base"
      :class="
        role === 'user'
          ? 'bg-teal-500 text-white rounded-tr-sm'
          : pending
            ? 'bg-teal-50/30 border border-teal-100 text-slate-500 rounded-tl-sm'
            : 'bg-white border border-gray-100 text-slate-700 rounded-tl-sm md-content'
      "
    >
      <ThinkingIndicator v-if="role === 'assistant' && pending" label="正在生成回复" />
      <div v-else-if="role === 'assistant'" v-html="html"></div>
      <template v-else>{{ content }}</template>
    </div>
  </div>
</template>

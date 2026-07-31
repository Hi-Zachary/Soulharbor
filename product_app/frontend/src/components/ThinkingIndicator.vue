<script setup>
defineProps({
  label: { type: String, default: "正在思考" },
});
</script>

<template>
  <span class="thinking-indicator" aria-live="polite">
    <span class="thinking-label">{{ label }}</span>
    <span class="thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span>
  </span>
</template>

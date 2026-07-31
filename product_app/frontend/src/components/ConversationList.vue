<script setup>
function fmtTime(ts) {
  try {
    const d = new Date(ts * 1000);
    return `${d.getMonth() + 1}-${d.getDate()} ${d.getHours().toString().padStart(2, "0")}:${d
      .getMinutes()
      .toString()
      .padStart(2, "0")}`;
  } catch (_) {
    return "";
  }
}

defineProps({
  conversations: { type: Array, default: () => [] },
});
const emit = defineEmits(["select", "rename", "remove", "refresh"]);

function onRename(c) {
  const title = (c.title || "").trim() || "新会话";
  const t = prompt("重命名会话：", title);
  if (t === null) return;
  emit("rename", c.sid, t);
}

function onRemove(c) {
  if (!confirm("确定要删除该会话吗？")) return;
  emit("remove", c.sid);
}
</script>

<template>
  <div
    class="lg:col-span-3 flex flex-col bg-white rounded-2xl border border-teal-100/50 shadow-sm overflow-hidden"
  >
    <div class="p-4 border-b border-gray-100 flex items-center justify-between bg-teal-50/30">
      <div class="text-sm font-bold text-teal-800">历史会话</div>
      <button
        class="text-xs text-teal-600 hover:text-teal-800 font-medium p-1 transition-colors"
        @click="emit('refresh')"
      >
        刷新
      </button>
    </div>
    <div class="flex-1 overflow-y-auto p-3 space-y-2 max-h-[70vh]">
      <div v-if="!conversations.length" class="text-xs text-center text-slate-400 py-4">
        暂无历史会话
      </div>
      <div
        v-for="c in conversations"
        :key="c.sid"
        class="rounded-xl border border-transparent bg-gray-50 hover:bg-teal-50/50 hover:border-teal-100 p-3 transition-colors group cursor-pointer"
        @click="emit('select', c.sid)"
      >
        <div class="flex flex-col gap-2">
          <div class="flex items-start justify-between">
            <div class="text-sm font-medium text-slate-700 truncate pr-2">
              {{ (c.title || "").trim() || "新会话" }}
            </div>
          </div>
          <div class="flex items-center justify-between">
            <div class="text-xs text-slate-400">{{ c.started_at ? fmtTime(c.started_at) : "" }}</div>
            <div class="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                class="text-xs px-2 py-1 rounded-md bg-white border border-gray-200 text-slate-600 hover:text-teal-600 hover:border-teal-200 shadow-sm"
                title="重命名"
                @click.stop="onRename(c)"
              >
                ✎
              </button>
              <button
                class="text-xs px-2 py-1 rounded-md bg-white border border-gray-200 text-rose-500 hover:bg-rose-50 hover:border-rose-200 shadow-sm"
                title="删除"
                @click.stop="onRemove(c)"
              >
                ✕
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

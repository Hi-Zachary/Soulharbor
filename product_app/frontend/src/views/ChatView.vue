<script setup>
import { nextTick, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import ChatBubble from "@/components/ChatBubble.vue";
import ChatInput from "@/components/ChatInput.vue";
import ConversationList from "@/components/ConversationList.vue";
import PrivacyModal from "@/components/PrivacyModal.vue";
import ThinkingIndicator from "@/components/ThinkingIndicator.vue";
import { useAuthStore } from "@/stores/auth";
import { useChatStore } from "@/stores/chat";
import { usePrivacyStore } from "@/stores/privacy";
import { renderAssistantMarkdown } from "@/components/markdown";

const router = useRouter();
const auth = useAuthStore();
const chat = useChatStore();
const privacy = usePrivacyStore();
const chatEl = ref(null);

async function openPrivacy() {
  privacy.open();
  const ok = await privacy.load();
  if (!ok) privacy.showToast("无法加载隐私设置");
}

async function onLogout() {
  await chat.logout();
  await auth.logout();
  router.push("/login");
}

watch(
  () => chat.messages.length + (chat.messages.at(-1)?.content?.length || 0),
  async () => {
    await nextTick();
    if (chatEl.value) chatEl.value.scrollTop = chatEl.value.scrollHeight;
  }
);

onMounted(async () => {
  const ok = await auth.refresh();
  if (!ok) {
    router.replace("/login");
    return;
  }
  chat.username = auth.username;
  await chat.loadHistory();
  await chat.loadConversations();
});
</script>

<template>
  <div class="bg-emerald-50 text-slate-800 font-sans min-h-screen flex flex-col">
    <div class="max-w-6xl w-full mx-auto px-4 py-6 flex-1 flex flex-col">
      <div class="flex items-center justify-between gap-3 mb-4">
        <div>
          <div class="text-2xl font-bold text-teal-700">SoulHarbor</div>
          <div class="text-sm text-teal-600/80 font-medium">校园心理支持助手</div>
        </div>
        <div class="flex items-center gap-3">
          <div class="text-sm text-slate-600 font-medium">你好，{{ auth.username || chat.username }}</div>
          <button
            class="px-4 py-2 rounded-xl bg-white hover:bg-gray-50 border border-gray-200 text-slate-600 text-sm font-medium shadow-sm transition-colors"
            @click="openPrivacy"
          >
            隐私
          </button>
          <button
            class="px-4 py-2 rounded-xl bg-teal-500 hover:bg-teal-600 text-white text-sm font-medium shadow-md shadow-teal-500/20 transition-colors"
            @click="chat.newConversation()"
          >
            新建会话
          </button>
          <button
            class="px-4 py-2 rounded-xl bg-white hover:bg-gray-50 border border-gray-200 text-slate-600 text-sm font-medium shadow-sm transition-colors"
            @click="onLogout"
          >
            退出
          </button>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">
        <ConversationList
          :conversations="chat.conversations"
          @select="chat.selectConversation"
          @rename="chat.renameConversation"
          @remove="chat.deleteConversation"
          @refresh="chat.loadConversations"
        />

        <div
          class="lg:col-span-9 flex flex-col bg-white rounded-2xl border border-teal-100/50 shadow-sm overflow-hidden"
        >
          <div
            ref="chatEl"
            class="flex-1 overflow-y-auto p-6 space-y-6 max-h-[60vh] bg-slate-50/50"
          >
            <div
              v-if="!chat.messages.length"
              class="flex flex-col items-center justify-center h-full text-slate-400 space-y-4 pt-10"
            >
              <svg
                class="w-16 h-16 text-teal-200"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="1.5"
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
              <p>今天想聊点什么呢？我在这里。</p>
            </div>

            <template v-for="(m, i) in chat.messages" :key="i">
              <div
                v-if="m.streaming"
                class="flex w-full justify-start"
              >
                <div
                  class="max-w-[85%] rounded-2xl px-5 py-3 leading-relaxed shadow-sm text-sm md:text-base border rounded-tl-sm"
                  :class="
                    m.content
                      ? 'bg-white border-gray-100 text-slate-700 md-content'
                      : 'border-teal-100 bg-teal-50/30 text-slate-700'
                  "
                >
                  <ThinkingIndicator v-if="!m.content" label="正在思考" />
                  <div v-else v-html="renderAssistantMarkdown(m.content)"></div>
                </div>
              </div>
              <ChatBubble
                v-else
                :role="m.role"
                :content="m.content"
                :pending="!!m.pending"
              />
            </template>
          </div>

          <ChatInput
            :disabled="chat.isStreaming"
            :sending-label="chat.sendingLabel"
            @send="chat.sendMessage"
          />
        </div>
      </div>
    </div>
    <PrivacyModal />
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import { useAuthStore } from "@/stores/auth";

const router = useRouter();
const auth = useAuthStore();
const username = ref("");
const password = ref("");
const err = ref("");

async function submit() {
  err.value = "";
  const j = await auth.register(username.value.trim(), password.value);
  if (!j.ok) {
    err.value = "注册失败：" + (j.error || "unknown");
    return;
  }
  router.push("/app");
}
</script>

<template>
  <div class="bg-emerald-50 text-slate-800 font-sans min-h-screen flex flex-col justify-center">
    <div class="max-w-md mx-auto px-4 py-12 w-full">
      <div class="text-center">
        <div class="text-3xl font-bold text-teal-700">SoulHarbor</div>
        <div class="text-sm text-slate-500 mt-2">在这里，有我们倾听。创建你的账号。</div>
      </div>

      <div class="mt-8 rounded-2xl border border-white bg-white/80 backdrop-blur shadow-xl shadow-teal-100/50 p-6">
        <div
          v-if="err"
          class="text-sm text-rose-500 bg-rose-50 p-3 rounded-lg mb-4 border border-rose-100"
        >
          {{ err }}
        </div>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-slate-700 mb-1">用户名</label>
            <input
              v-model="username"
              class="w-full rounded-xl bg-gray-50 border border-gray-200 px-4 py-2.5 outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 transition-all"
              placeholder="建议英文/数字"
              @keydown.enter="submit"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-slate-700 mb-1">密码</label>
            <input
              v-model="password"
              type="password"
              class="w-full rounded-xl bg-gray-50 border border-gray-200 px-4 py-2.5 outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 transition-all"
              placeholder="至少 6 位"
              @keydown.enter="submit"
            />
          </div>
          <button
            class="w-full mt-2 px-4 py-3 rounded-xl bg-teal-500 hover:bg-teal-600 text-white font-medium shadow-md shadow-teal-500/30 transition-all transform hover:-translate-y-0.5"
            @click="submit"
          >
            注 册
          </button>
        </div>
        <div class="mt-6 text-center text-sm text-slate-500">
          已有账号？
          <router-link
            class="text-teal-600 hover:text-teal-700 font-medium ml-1 underline underline-offset-2"
            to="/login"
          >
            直接登录
          </router-link>
        </div>
      </div>
    </div>
  </div>
</template>

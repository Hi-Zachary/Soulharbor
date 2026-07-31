<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import { postJson } from "@/api/client";

const router = useRouter();
const password = ref("");
const err = ref("");

async function submit() {
  err.value = "";
  const j = await postJson("/admin/api/login", { password: password.value });
  if (!j.ok) {
    err.value = "登录失败：" + (j.error || "unknown");
    return;
  }
  // Admin dashboard remains the Jinja template for full ops UI.
  window.location.href = "/admin";
}
</script>

<template>
  <div class="bg-emerald-50 text-slate-800 font-sans min-h-screen flex flex-col justify-center">
    <div class="max-w-md mx-auto px-4 py-12 w-full">
      <div class="text-center">
        <div class="text-3xl font-bold text-teal-700">SoulHarbor 后台</div>
        <div class="text-sm text-slate-500 mt-2">仅用于匿名统计与运营</div>
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
            <label class="block text-sm font-medium text-slate-700 mb-1">管理员口令</label>
            <input
              v-model="password"
              type="password"
              class="w-full rounded-xl bg-gray-50 border border-gray-200 px-4 py-2.5 outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 transition-all"
              placeholder="输入口令"
              @keydown.enter="submit"
            />
          </div>
          <button
            class="w-full mt-2 px-4 py-3 rounded-xl bg-teal-500 hover:bg-teal-600 text-white font-medium shadow-md shadow-teal-500/30 transition-all transform hover:-translate-y-0.5"
            @click="submit"
          >
            登 录
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

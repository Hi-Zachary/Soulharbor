import { createRouter, createWebHistory } from "vue-router";
import ChatView from "@/views/ChatView.vue";
import LoginView from "@/views/LoginView.vue";
import RegisterView from "@/views/RegisterView.vue";
import AdminLoginView from "@/views/AdminLoginView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/app" },
    { path: "/app", name: "chat", component: ChatView },
    { path: "/login", name: "login", component: LoginView },
    { path: "/register", name: "register", component: RegisterView },
    { path: "/admin/login", name: "admin-login", component: AdminLoginView },
  ],
});

export default router;

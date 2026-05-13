import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api",
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (!token) {
    return config;
  }
  const value = `Bearer ${token}`;
  // Axios v1 uses AxiosHeaders; assignment on .Authorization can be ignored without .set().
  if (config.headers && typeof config.headers.set === "function") {
    config.headers.set("Authorization", value);
  } else {
    config.headers = { ...(config.headers || {}), Authorization: value };
  }
  return config;
});

/** Set from main app once the router exists (avoids importing router into api). */
let onAuthRequired = null;

export function setAuthRequiredHandler(handler) {
  onAuthRequired = typeof handler === "function" ? handler : null;
}

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    if (status === 401) {
      const hadToken = Boolean(localStorage.getItem("token"));
      if (hadToken) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        try {
          onAuthRequired?.();
        } catch {
          /* ignore */
        }
      }
    }
    return Promise.reject(error);
  }
);

export default api;

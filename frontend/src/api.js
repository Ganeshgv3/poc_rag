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

export default api;

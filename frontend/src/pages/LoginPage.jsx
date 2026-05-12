import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import api from "../api";
import AuthCard from "../components/AuthCard";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const onSubmit = async (event) => {
    event.preventDefault();
    setError("");
    try {
      const { data } = await api.post("/auth/login", { email, password });
      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));
      localStorage.setItem("startFreshChat", "1");
      navigate("/chat");
    } catch (err) {
      setError(err.response?.data?.error || err.response?.data?.detail || "Login failed.");
    }
  };

  return (
    <AuthCard title="Welcome back" subtitle="Step into your premium AI cockpit for PDF intelligence.">
      <form onSubmit={onSubmit} className="auth-form">
        <input value={email} onChange={(e) => setEmail(e.target.value)} type="email" placeholder="Email" required />
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          type="password"
          placeholder="Password"
          required
        />
        {error && <div className="error-text">{error}</div>}
        <button type="submit">Enter Workspace</button>
      </form>
      <p className="auth-link">
        No account? <Link to="/register">Create one</Link>
      </p>
    </AuthCard>
  );
}

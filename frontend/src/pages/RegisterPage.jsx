import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import api from "../api";
import AuthCard from "../components/AuthCard";

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const onSubmit = async (event) => {
    event.preventDefault();
    setError("");
    try {
      const { data } = await api.post("/auth/register", { name, email, password });
      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));
      navigate("/chat");
    } catch (err) {
      setError(err.response?.data?.error || err.response?.data?.detail || "Register failed.");
    }
  };

  return (
    <AuthCard title="Create account" subtitle="Launch your ultra-creative AI knowledge workspace in seconds.">
      <form onSubmit={onSubmit} className="auth-form">
        <input value={name} onChange={(e) => setName(e.target.value)} type="text" placeholder="Full name" required />
        <input value={email} onChange={(e) => setEmail(e.target.value)} type="email" placeholder="Email" required />
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          type="password"
          placeholder="Password (min 6)"
          required
          minLength={6}
        />
        {error && <div className="error-text">{error}</div>}
        <button type="submit">Create Premium Access</button>
      </form>
      <p className="auth-link">
        Already have an account? <Link to="/login">Login</Link>
      </p>
    </AuthCard>
  );
}

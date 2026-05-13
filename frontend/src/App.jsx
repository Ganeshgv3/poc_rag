import { useEffect } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { setAuthRequiredHandler } from "./api";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import ChatPage from "./pages/ChatPage";

function ProtectedRoute({ children }) {
  const token = localStorage.getItem("token");
  return token ? children : <Navigate to="/login" replace />;
}

function AuthSessionSync() {
  const navigate = useNavigate();
  useEffect(() => {
    setAuthRequiredHandler(() => navigate("/login", { replace: true }));
    return () => setAuthRequiredHandler(null);
  }, [navigate]);
  return null;
}

export default function App() {
  return (
    <>
      <AuthSessionSync />
      <Routes>
      <Route path="/" element={<Navigate to="/chat" replace />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route
        path="/chat"
        element={
          <ProtectedRoute>
            <ChatPage />
          </ProtectedRoute>
        }
      />
    </Routes>
    </>
  );
}

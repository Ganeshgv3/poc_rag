export default function AuthCard({ title, subtitle, children }) {
  return (
    <div className="auth-shell">
      <div className="auth-bg-orb auth-bg-orb-one" />
      <div className="auth-bg-orb auth-bg-orb-two" />
      <div className="auth-card">
        <div className="auth-badge">Premium AI Workspace</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
        {children}
      </div>
    </div>
  );
}

import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./shared/hooks/use-auth";
import useAuth from "./shared/hooks/use-auth";
import TopNav from "./components/layout/top-nav";
import HomePage from "./components/HomePage";
import ComposePage from "./components/ComposePage";
import SignInPage from "./components/SignInPage";

function RequireAuth({ children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/signin" replace />;
  return children;
}

function RedirectIfAuth({ children }) {
  const { user } = useAuth();
  if (user) return <Navigate to="/compose-search" replace />;
  return children;
}

function AppRoutes() {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-slate-50">
      {user && <TopNav />}
      <Routes>
        <Route path="/signin" element={<RedirectIfAuth><SignInPage /></RedirectIfAuth>} />
        <Route path="/" element={<RequireAuth><HomePage /></RequireAuth>} />
        <Route path="/compose-search" element={<RequireAuth><ComposePage /></RequireAuth>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </Router>
  );
}

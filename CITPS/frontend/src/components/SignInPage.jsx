import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { LogIn, UserPlus } from "lucide-react";
import useAuth from "@/shared/hooks/use-auth";
import Button from "./ui/button";
import PageTransition from "./layout/page-transition";

export default function SignInPage() {
  const { signin, signup, loading, error, clearError } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState("signin"); // "signin" | "signup"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [localError, setLocalError] = useState(null);

  const toggleMode = () => {
    setMode((m) => (m === "signin" ? "signup" : "signin"));
    setLocalError(null);
    clearError();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError(null);
    clearError();

    if (!username.trim() || !password) {
      setLocalError("Username and password are required.");
      return;
    }

    if (mode === "signup") {
      if (password !== confirmPassword) {
        setLocalError("Passwords do not match.");
        return;
      }
      if (password.length < 4) {
        setLocalError("Password must be at least 4 characters.");
        return;
      }
    }

    try {
      if (mode === "signin") {
        await signin(username.trim(), password);
      } else {
        await signup(username.trim(), password);
      }
      navigate("/compose-search");
    } catch {
      // error is already in auth context
    }
  };

  const displayError = localError || error;

  return (
    <PageTransition>
      <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
        <div className="w-full max-w-sm">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-primary">CITPS</h1>
            <p className="text-sm text-text-secondary mt-1">
              Composed Image‑Text Person Search
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-card border border-border p-6">
            <h2 className="text-xl font-semibold text-text-primary mb-6 text-center">
              {mode === "signin" ? "Sign In" : "Create Account"}
            </h2>

            {displayError && (
              <div className="mb-4 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
                {displayError}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-text-secondary mb-1">
                  Username
                </label>
                <input
                  id="username"
                  type="text"
                  autoComplete="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full rounded-xl border border-border bg-white px-3 py-2.5 text-sm text-text-primary placeholder:text-text-secondary/50 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary"
                  placeholder="Enter your username"
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-text-secondary mb-1">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  autoComplete={mode === "signin" ? "current-password" : "new-password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-xl border border-border bg-white px-3 py-2.5 text-sm text-text-primary placeholder:text-text-secondary/50 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary"
                  placeholder="Enter your password"
                />
              </div>

              {mode === "signup" && (
                <div>
                  <label htmlFor="confirm-password" className="block text-sm font-medium text-text-secondary mb-1">
                    Confirm Password
                  </label>
                  <input
                    id="confirm-password"
                    type="password"
                    autoComplete="new-password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full rounded-xl border border-border bg-white px-3 py-2.5 text-sm text-text-primary placeholder:text-text-secondary/50 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary"
                    placeholder="Re-enter your password"
                  />
                </div>
              )}

              <Button
                type="submit"
                loading={loading}
                icon={mode === "signin" ? <LogIn className="h-4 w-4" /> : <UserPlus className="h-4 w-4" />}
                className="w-full"
              >
                {mode === "signin" ? "Sign In" : "Sign Up"}
              </Button>
            </form>

            <div className="mt-5 text-center">
              <button
                type="button"
                onClick={toggleMode}
                className="text-sm text-primary hover:text-primary-hover font-medium cursor-pointer"
              >
                {mode === "signin"
                  ? "Don't have an account? Sign Up"
                  : "Already have an account? Sign In"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}


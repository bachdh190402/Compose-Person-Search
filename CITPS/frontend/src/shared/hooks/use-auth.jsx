import { createContext, useContext, useState, useCallback, useMemo } from "react";
import { API_BASE_URL } from "@/shared/config";

const AuthContext = createContext(null);

const STORAGE_KEY = "citps_user";

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const signin = useCallback(async (username, password) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/auth/signin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Sign in failed (${res.status})`);
      }
      const data = await res.json();
      const userObj = { username: data.username };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(userObj));
      setUser(userObj);
      return userObj;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const signup = useCallback(async (username, password) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Sign up failed (${res.status})`);
      }
      const data = await res.json();
      const userObj = { username: data.username };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(userObj));
      setUser(userObj);
      return userObj;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const signout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
    setError(null);
  }, []);

  const clearError = useCallback(() => setError(null), []);

  const value = useMemo(
    () => ({ user, loading, error, signin, signup, signout, clearError }),
    [user, loading, error, signin, signup, signout, clearError],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export default function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider>");
  return ctx;
}


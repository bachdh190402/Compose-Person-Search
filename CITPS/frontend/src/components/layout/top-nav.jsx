import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Menu, LogOut } from "lucide-react";
import MobileMenu from "./mobile-menu";
import useAuth from "@/shared/hooks/use-auth";

const NAV_LINKS = [
  { path: "/", label: "Home" },
  { path: "/compose-search", label: "Compose Search" },
];

export default function TopNav() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const { user, signout } = useAuth();

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-border bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <Link to="/" className="flex items-center gap-2">
            <span className="text-lg font-bold text-primary">CITPS</span>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {NAV_LINKS.map((link) => {
              const isActive = location.pathname === link.path;
              return (
                <Link
                  key={link.label}
                  to={link.path}
                  className="relative px-3 py-2 text-sm font-medium text-text-secondary hover:text-text-primary transition-colors"
                >
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="absolute inset-x-1 -bottom-[1px] h-0.5 bg-primary rounded-full"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                    />
                  )}
                  <span className={isActive ? "text-text-primary" : ""}>{link.label}</span>
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-3">
            {user && (
              <>
                <span className="hidden sm:inline text-sm text-text-secondary">
                  Hi, <strong className="text-text-primary">{user.username}</strong>
                </span>
                <button
                  onClick={signout}
                  className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-text-secondary hover:text-error hover:bg-red-50 transition-colors cursor-pointer"
                  title="Sign out"
                >
                  <LogOut className="h-4 w-4" />
                  <span className="hidden sm:inline">Sign Out</span>
                </button>
              </>
            )}

            <button
              className="md:hidden p-2 text-text-secondary hover:text-text-primary cursor-pointer"
              onClick={() => setMobileOpen(true)}
            >
              <Menu className="h-5 w-5" />
            </button>
          </div>
        </div>
      </header>

      <MobileMenu isOpen={mobileOpen} onClose={() => setMobileOpen(false)} links={NAV_LINKS} />
    </>
  );
}

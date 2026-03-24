import { Link, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

export default function MobileMenu({ isOpen, onClose, links }) {
  const location = useLocation();

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-50 bg-black/20 backdrop-blur-sm"
          />

          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", bounce: 0.1, duration: 0.4 }}
            className="fixed right-0 top-0 z-50 h-full w-72 bg-white shadow-modal p-6"
          >
            <div className="flex items-center justify-between mb-8">
              <span className="text-lg font-semibold text-text-primary">Menu</span>
              <button onClick={onClose} className="p-2 text-text-secondary hover:text-text-primary cursor-pointer">
                <X className="h-5 w-5" />
              </button>
            </div>

            <nav className="flex flex-col gap-1">
              {links.map((link) => {
                const isActive = location.pathname === link.path;
                return (
                  <Link
                    key={link.label}
                    to={link.path}
                    onClick={onClose}
                    className={cn(
                      "rounded-xl px-4 py-3 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-blue-50 text-primary"
                        : "text-text-secondary hover:bg-slate-50 hover:text-text-primary"
                    )}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </nav>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

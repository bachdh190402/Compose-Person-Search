import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, XCircle, X } from "lucide-react";
import { cn } from "@/lib/utils";

const icons = {
  success: CheckCircle2,
  error: XCircle,
};

const styles = {
  success: "bg-green-50 border-green-200 text-green-800",
  error: "bg-red-50 border-red-200 text-red-800",
};

/**
 * Toast notification – auto-dismisses after `duration` ms.
 *
 * Props:
 *   open       – boolean
 *   type       – "success" | "error"
 *   title      – string
 *   message    – string (optional)
 *   duration   – ms (default 5000, 0 = manual close)
 *   onClose    – () => void
 */
export default function Toast({ open, type = "success", title, message, duration = 5000, onClose }) {
  useEffect(() => {
    if (!open || duration <= 0) return;
    const timer = setTimeout(onClose, duration);
    return () => clearTimeout(timer);
  }, [open, duration, onClose]);

  const Icon = icons[type] || icons.success;

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: -20, x: "-50%" }}
          animate={{ opacity: 1, y: 0, x: "-50%" }}
          exit={{ opacity: 0, y: -20, x: "-50%" }}
          transition={{ type: "spring", bounce: 0.3, duration: 0.4 }}
          className={cn(
            "fixed top-20 left-1/2 z-[100] flex items-start gap-3 rounded-xl border px-5 py-4 shadow-lg max-w-md w-[90vw]",
            styles[type],
          )}
        >
          <Icon className="h-5 w-5 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold">{title}</p>
            {message && <p className="text-sm mt-0.5 opacity-80">{message}</p>}
          </div>
          <button onClick={onClose} className="shrink-0 p-1 rounded-lg hover:bg-black/5 cursor-pointer transition-colors">
            <X className="h-4 w-4" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}


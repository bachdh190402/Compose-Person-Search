import { motion } from "framer-motion";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export default function ChipSelector({ label, options, value, onChange, multiSelect = false }) {
  const selected = Array.isArray(value) ? value : [value];

  const handleSelect = (option) => {
    if (multiSelect) {
      const next = selected.includes(option)
        ? selected.filter((v) => v !== option)
        : [...selected, option];
      onChange(next);
    } else {
      onChange(option);
    }
  };

  return (
    <div className="space-y-2">
      {label && <p className="text-sm font-medium text-text-secondary">{label}</p>}
      <div className="flex flex-wrap gap-2">
        {options.map((option) => {
          const isActive = selected.includes(option);
          return (
            <motion.button
              key={option}
              type="button"
              whileTap={{ scale: 0.95 }}
              onClick={() => handleSelect(option)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium transition-colors cursor-pointer",
                isActive
                  ? "bg-primary text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              )}
            >
              {isActive && <Check className="h-3.5 w-3.5" />}
              {option}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}

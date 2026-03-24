import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { SlidersHorizontal, X } from "lucide-react";
import { cn } from "@/lib/utils";

export default function SplitLayout({ sidebar, children, sidebarTitle = "Filters" }) {
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);

  return (
    <div className="flex gap-6 min-h-[calc(100vh-8rem)]">
      <aside className="hidden lg:block w-80 shrink-0">
        <div className="sticky top-24 space-y-6">{sidebar}</div>
      </aside>

      <main className="flex-1 min-w-0">{children}</main>

      <button
        onClick={() => setMobileFiltersOpen(true)}
        className={cn(
          "lg:hidden fixed bottom-6 right-6 z-40 flex items-center gap-2",
          "rounded-full bg-primary text-white px-4 py-3 shadow-lg cursor-pointer",
          "hover:bg-primary-hover transition-colors"
        )}
      >
        <SlidersHorizontal className="h-4 w-4" />
        <span className="text-sm font-medium">{sidebarTitle}</span>
      </button>

      <AnimatePresence>
        {mobileFiltersOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileFiltersOpen(false)}
              className="lg:hidden fixed inset-0 z-50 bg-black/20 backdrop-blur-sm"
            />
            <motion.div
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", bounce: 0.1, duration: 0.4 }}
              className="lg:hidden fixed left-0 top-0 z-50 h-full w-80 bg-white shadow-modal overflow-y-auto"
            >
              <div className="flex items-center justify-between p-4 border-b border-border">
                <span className="text-lg font-semibold text-text-primary">{sidebarTitle}</span>
                <button
                  onClick={() => setMobileFiltersOpen(false)}
                  className="p-2 text-text-secondary hover:text-text-primary cursor-pointer"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <div className="p-4 space-y-6">{sidebar}</div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

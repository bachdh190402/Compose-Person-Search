import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import LazyImage from "./lazy-image";

export default function ImageCard({ src, alt, score, index, onClick, children, className }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      onClick={onClick}
      className={cn(
        "group relative overflow-hidden rounded-xl border border-border bg-white shadow-card hover:shadow-hover transition-shadow",
        onClick && "cursor-pointer",
        className
      )}
    >
      <div className="absolute top-2 left-2 z-10 rounded-full bg-black/50 px-2 py-1 text-xs font-medium text-white">
        {(index ?? 0) + 1}
      </div>

      <LazyImage
        src={src}
        alt={alt || `Result ${(index ?? 0) + 1}`}
        className="aspect-[3/4]"
      />

      {score !== undefined && (
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="absolute bottom-2 left-2">
            <span className="text-xs font-medium text-white bg-black/40 px-2 py-1 rounded-full">
              Score: {typeof score === "number" ? score.toFixed(3) : score}
            </span>
          </div>
        </div>
      )}

      {children && <div className="p-3">{children}</div>}
    </motion.div>
  );
}

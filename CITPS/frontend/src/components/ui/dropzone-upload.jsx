import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X } from "lucide-react";
import { cn } from "@/lib/utils";

export default function DropzoneUpload({ onFileSelect, preview, onRemove, accept = { "image/*": [] }, maxSize = 10485760 }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept,
    maxSize,
    multiple: false,
    onDrop: (accepted) => {
      if (accepted.length > 0) onFileSelect(accepted[0]);
    },
  });

  return (
    <div className="w-full">
      <AnimatePresence mode="wait">
        {preview ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative group"
          >
            <img
              src={preview}
              alt="Upload preview"
              className="w-full rounded-xl object-contain max-h-64 border border-border"
            />
            <button
              type="button"
              onClick={onRemove}
              className="absolute top-2 right-2 p-1.5 rounded-full bg-white/90 text-slate-600 hover:bg-white hover:text-error transition-colors opacity-0 group-hover:opacity-100 cursor-pointer"
            >
              <X className="h-4 w-4" />
            </button>
          </motion.div>
        ) : (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            {...getRootProps()}
            className={cn(
              "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 transition-colors cursor-pointer",
              isDragActive
                ? "border-primary bg-blue-50"
                : "border-slate-300 hover:border-primary/50 hover:bg-slate-50"
            )}
          >
            <input {...getInputProps()} />
            <Upload className={cn("h-8 w-8", isDragActive ? "text-primary" : "text-slate-400")} />
            <div className="text-center">
              <p className="text-sm font-medium text-text-primary">
                {isDragActive ? "Drop image here" : "Drop image here"}
              </p>
              <p className="text-xs text-text-secondary mt-1">or click to browse</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

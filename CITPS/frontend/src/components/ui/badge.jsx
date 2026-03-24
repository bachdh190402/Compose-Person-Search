import { cn } from "@/lib/utils";

const variants = {
  default: "bg-slate-100 text-slate-700",
  primary: "bg-blue-50 text-blue-700",
  success: "bg-green-50 text-green-700",
  warning: "bg-amber-50 text-amber-700",
  error: "bg-red-50 text-red-700",
};

export default function Badge({ className, variant = "default", children, ...props }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}

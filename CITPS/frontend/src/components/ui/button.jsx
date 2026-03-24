import { forwardRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import Spinner from "./spinner";

const variants = {
  primary: "bg-primary text-white hover:bg-primary-hover",
  secondary: "bg-slate-100 text-slate-900 hover:bg-slate-200",
  outline: "border border-border bg-white text-slate-900 hover:bg-slate-50",
  ghost: "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
  danger: "bg-error text-white hover:bg-red-700",
};

const sizes = {
  sm: "h-8 px-3 text-sm rounded-lg",
  md: "h-10 px-4 text-sm rounded-xl",
  lg: "h-12 px-6 text-base rounded-xl",
};

const Button = forwardRef(({ className, variant = "primary", size = "md", loading, disabled, icon, children, ...props }, ref) => {
  const isDisabled = disabled || loading;
  return (
    <motion.button
      ref={ref}
      whileTap={!isDisabled ? { scale: 0.97 } : undefined}
      className={cn(
        "inline-flex items-center justify-center gap-2 font-medium transition-colors cursor-pointer",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        variants[variant],
        sizes[size],
        className
      )}
      disabled={isDisabled}
      {...props}
    >
      {loading ? <Spinner size="sm" /> : icon ? icon : null}
      {children}
    </motion.button>
  );
});

Button.displayName = "Button";
export default Button;

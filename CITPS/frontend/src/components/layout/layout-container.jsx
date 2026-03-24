import { cn } from "@/lib/utils";

export default function LayoutContainer({ className, children, ...props }) {
  return (
    <div className={cn("mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8", className)} {...props}>
      {children}
    </div>
  );
}

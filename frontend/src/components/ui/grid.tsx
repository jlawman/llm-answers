import React from 'react'
import { cn } from "@/lib/utils"

interface GridProps extends React.HTMLAttributes<HTMLDivElement> {
  columns?: number
  gap?: number
  className?: string
  children: React.ReactNode
}

export const Grid = React.forwardRef<HTMLDivElement, GridProps>(
  ({ columns = 1, gap = 4, className, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "grid",
          `grid-cols-${columns}`,
          `gap-${gap}`,
          className
        )}
        {...props}
      >
        {children}
      </div>
    )
  }
)

Grid.displayName = "Grid"

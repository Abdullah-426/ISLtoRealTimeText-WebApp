'use client';

import * as React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface GlowingEffectProps extends React.HTMLAttributes<HTMLDivElement> {
    children: React.ReactNode;
    intensity?: 'low' | 'medium' | 'high';
    color?: 'blue' | 'purple' | 'green' | 'pink' | 'orange';
    size?: 'sm' | 'md' | 'lg' | 'xl';
}

const GlowingEffect = React.forwardRef<HTMLDivElement, GlowingEffectProps>(
    ({ children, intensity = 'medium', color = 'blue', size = 'md', className, ...props }, ref) => {
        const intensityClasses = {
            low: 'shadow-blue-500/20',
            medium: 'shadow-blue-500/40',
            high: 'shadow-blue-500/60'
        };

        const colorClasses = {
            blue: 'shadow-blue-500/40',
            purple: 'shadow-purple-500/40',
            green: 'shadow-green-500/40',
            pink: 'shadow-pink-500/40',
            orange: 'shadow-orange-500/40'
        };

        const sizeClasses = {
            sm: 'shadow-lg',
            md: 'shadow-xl',
            lg: 'shadow-2xl',
            xl: 'shadow-3xl'
        };

        return (
            <motion.div
                ref={ref}
                className={cn(
                    "relative",
                    sizeClasses[size],
                    colorClasses[color],
                    className
                )}
                whileHover={{
                    scale: 1.02,
                    boxShadow: "0 0 30px rgba(59, 130, 246, 0.5)"
                }}
                transition={{ duration: 0.3 }}
            >
                {/* Glow effect */}
                <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-500/20 to-purple-500/20 blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                {/* Content */}
                <div className="relative z-10">
                    {children}
                </div>
            </motion.div>
        );
    }
);

GlowingEffect.displayName = "GlowingEffect";

export { GlowingEffect };

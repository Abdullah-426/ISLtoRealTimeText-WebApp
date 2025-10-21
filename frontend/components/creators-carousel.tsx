'use client';

import * as React from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Github, Linkedin, Twitter } from "lucide-react";
import { cn } from "@/lib/utils";

// Define the type for a creator
export interface Creator {
    id: string;
    name: string;
    role: string;
    avatar: string;
    description: string;
    social: {
        github?: string;
        linkedin?: string;
        twitter?: string;
    };
}

// Props for the CreatorCard component
interface CreatorCardProps {
    creator: Creator;
}

// The individual creator card component
const CreatorCard = React.forwardRef<HTMLDivElement, CreatorCardProps>(({ creator }, ref) => (
    <motion.div
        ref={ref}
        className="relative flex-shrink-0 w-[220px] h-[260px] rounded-2xl overflow-hidden group snap-start bg-black/50 backdrop-blur-sm border-2 border-white/20"
        whileHover={{ y: -8, scale: 1.02 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
    >
        {/* Card Content */}
        <div className="relative z-10 p-6 h-full flex flex-col justify-center items-center">
            {/* Avatar */}
            <div className="flex justify-center mb-4">
                <div className="w-16 h-16 rounded-full bg-slate-700 border border-slate-600 flex items-center justify-center text-slate-300 text-xl font-bold">
                    {creator.name.split(' ').map(n => n[0]).join('')}
                </div>
            </div>

            {/* Creator Info */}
            <div className="text-center space-y-2">
                <h3 className="text-lg font-bold text-white">{creator.name}</h3>
                {creator.role === 'Lead Developer' && (
                    <p className="text-sm text-blue-300 font-medium">{creator.role}</p>
                )}
            </div>

            {/* Social Links */}
            <div className="flex justify-center gap-3 pt-4">
                {creator.social.github && (
                    <a
                        href={creator.social.github}
                        className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-slate-300 hover:bg-slate-600 hover:text-white transition-colors"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        <Github className="w-4 h-4" />
                    </a>
                )}
                {creator.social.linkedin && (
                    <a
                        href={creator.social.linkedin}
                        className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-slate-300 hover:bg-slate-600 hover:text-white transition-colors"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        <Linkedin className="w-4 h-4" />
                    </a>
                )}
            </div>
        </div>
    </motion.div>
));
CreatorCard.displayName = "CreatorCard";

// Props for the CreatorsCarousel component
export interface CreatorsCarouselProps extends React.HTMLAttributes<HTMLDivElement> {
    creators: Creator[];
}

// The main carousel component
const CreatorsCarousel = React.forwardRef<HTMLDivElement, CreatorsCarouselProps>(
    ({ creators, className, ...props }, ref) => {
        const scrollContainerRef = React.useRef<HTMLDivElement>(null);

        const scroll = (direction: "left" | "right") => {
            if (scrollContainerRef.current) {
                const { current } = scrollContainerRef;
                const scrollAmount = current.clientWidth * 0.8;
                current.scrollBy({
                    left: direction === "left" ? -scrollAmount : scrollAmount,
                    behavior: "smooth",
                });
            }
        };

        return (
            <div ref={ref} className={cn("relative w-full max-w-7xl mx-auto", className)} {...props}>
                {/* Container for all cards */}
                <div
                    ref={scrollContainerRef}
                    className="flex space-x-6 pb-4 justify-center flex-wrap gap-6"
                >
                    {creators.map((creator) => (
                        <CreatorCard key={creator.id} creator={creator} />
                    ))}
                </div>
            </div>
        );
    }
);
CreatorsCarousel.displayName = "CreatorsCarousel";

export { CreatorsCarousel, CreatorCard };

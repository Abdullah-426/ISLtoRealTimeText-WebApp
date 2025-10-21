'use client';
import { useStore } from '@/lib/store';
import { motion, AnimatePresence } from 'framer-motion';

export function TopK() {
    const topk = useStore((s: any) => s.topk);
    // Only show the top 3 predictions
    const topThree = topk.slice(0, 3);

    return (
        <div className='bg-black/50 backdrop-blur-sm border border-white/10 p-6 rounded-2xl shadow-sm'>
            <h2 className='font-display text-lg mb-3'>Top 3</h2>
            <div className='flex gap-2 justify-start items-center min-h-[2.5rem]'>
                <AnimatePresence mode="wait">
                    {topThree.map((t: any, index: number) => (
                        <motion.span
                            key={`${t.label}-${index}`}
                            initial={{ opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -4 }}
                            transition={{ duration: 0.2 }}
                            className='inline-flex items-center px-3 py-1 rounded-full text-sm bg-white/5 border border-white/10 flex-shrink-0 min-w-[80px]'>
                            <span className='font-medium mr-2 truncate'>{t.label}</span>
                            <span className='text-gray-400'>{Math.round(t.prob * 100)}%</span>
                        </motion.span>
                    ))}
                </AnimatePresence>
            </div>
        </div>
    );
}

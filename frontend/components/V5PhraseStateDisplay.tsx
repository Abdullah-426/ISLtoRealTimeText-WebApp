'use client';
import { useStore } from '@/lib/store';
import { CircularProgress } from './CircularProgress';
import { motion, AnimatePresence } from 'framer-motion';
import { useMemo, useEffect, useState } from 'react';

export function V5PhraseStateDisplay() {
    const mode = useStore((s: any) => s.mode);
    const v5_state = useStore((s: any) => s.v5_state);
    const v5_bothHandsCount = useStore((s: any) => s.v5_bothHandsCount);
    const v5_start_hold_frames = useStore((s: any) => s.v5_start_hold_frames);
    const v5_segmentStartTime = useStore((s: any) => s.v5_segmentStartTime);
    const v5_cooldownStartTime = useStore((s: any) => s.v5_cooldownStartTime);
    const v5_frames_per_segment = useStore((s: any) => s.v5_frames_per_segment);
    const v5_target_fps = useStore((s: any) => s.v5_target_fps);
    const v5_segment_cooldown = useStore((s: any) => s.v5_segment_cooldown);
    const topk = useStore((s: any) => s.topk);

    // Use state for smooth updates
    const [currentTime, setCurrentTime] = useState(Date.now());

    // Update time every 50ms for very smooth progress
    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentTime(Date.now());
        }, 50);
        return () => clearInterval(interval);
    }, []);

    // Memoize calculations to prevent unnecessary re-renders
    const { progress, statusText, progressColor, showProgress } = useMemo(() => {
        let progress = 0;
        let statusText = '';
        let progressColor = '#10b981'; // green
        let showProgress = false;

        if (v5_state === 'wait_start') {
            progress = v5_bothHandsCount / v5_start_hold_frames;
            statusText = 'SHOW BOTH HANDS to start';
            progressColor = '#10b981'; // green
            showProgress = true;
        } else if (v5_state === 'capture') {
            const elapsed = (currentTime - v5_segmentStartTime) / 1000;
            const segmentDuration = v5_frames_per_segment / v5_target_fps;
            progress = Math.min(1, elapsed / segmentDuration);
            statusText = `CAPTURE: ${elapsed.toFixed(1)}s / ${segmentDuration.toFixed(1)}s`;
            progressColor = '#10b981'; // green
            showProgress = true;
        } else if (v5_state === 'predict') {
            statusText = 'PREDICTING...';
            progressColor = '#f59e0b'; // amber
            showProgress = false;
        } else if (v5_state === 'cooldown') {
            const elapsed = (currentTime - v5_cooldownStartTime) / 1000;
            const remain = Math.max(0, v5_segment_cooldown - elapsed);
            progress = (v5_segment_cooldown - remain) / v5_segment_cooldown;
            statusText = `COOLDOWN: ${remain.toFixed(1)}s`;
            progressColor = '#f97316'; // orange
            showProgress = true;
        }

        return { progress, statusText, progressColor, showProgress };
    }, [v5_state, v5_bothHandsCount, v5_start_hold_frames, v5_segmentStartTime, v5_cooldownStartTime, v5_frames_per_segment, v5_target_fps, v5_segment_cooldown, currentTime]);

    // Only show for phrases mode
    if (mode !== 'phrases') {
        return null;
    }

    return (
        <div className='bg-black/50 backdrop-blur-sm border border-white/10 p-6 rounded-2xl shadow-sm'>
            <h2 className='font-display text-lg mb-4'>Phrase Recognition</h2>

            {/* State Display */}
            <div className='flex items-center justify-center mb-4'>
                {showProgress ? (
                    <CircularProgress
                        progress={progress}
                        size={100}
                        strokeWidth={6}
                        color={progressColor}
                        backgroundColor="#374151"
                    >
                        <div className='text-center'>
                            <div className='text-xs text-gray-400'>{Math.round(progress * 100)}%</div>
                        </div>
                    </CircularProgress>
                ) : (
                    <div className='w-[100px] h-[100px] rounded-full border-4 border-amber-500 flex items-center justify-center'>
                        <div className='text-xs text-amber-500 font-medium'>PREDICT</div>
                    </div>
                )}
            </div>

            {/* Status Text */}
            <div className='text-center mb-4'>
                <motion.div
                    key={statusText}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3 }}
                    className='text-sm font-medium text-gray-300'
                >
                    {statusText}
                </motion.div>
            </div>

            {/* Top 3 Predictions */}
            <div className='space-y-2'>
                <h3 className='text-sm font-medium text-gray-400'>Top Predictions:</h3>
                <AnimatePresence mode="wait">
                    {topk && topk.length > 0 ? (
                        topk.slice(0, 3).map((pred: any, index: number) => (
                            <motion.div
                                key={`${pred.label}-${index}`}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                transition={{ duration: 0.3, delay: index * 0.1 }}
                                className='flex items-center justify-between p-2 bg-white/5 rounded-lg'
                            >
                                <span className='text-sm font-medium text-white'>{pred.label}</span>
                                <span className='text-xs text-gray-400'>{Math.round(pred.prob * 100)}%</span>
                            </motion.div>
                        ))
                    ) : (
                        <div className='text-sm text-gray-500 text-center py-2'>
                            No predictions yet
                        </div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}

'use client';
import { useEffect } from 'react';
import { useStore } from '@/lib/store';

export function CommitBar() {
    const holdProgress = useStore((s: any) => s.holdProgress);
    const toggleHold = useStore((s: any) => s.toggleHold);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.code === 'Space') { e.preventDefault(); toggleHold(); }
            if (e.ctrlKey && e.key === 'Enter') { useStore.getState().commitNow(); }
            if (e.ctrlKey && (e.key.toLowerCase() === 'z')) { useStore.getState().undo(); }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [toggleHold]);

    const offset = 100 - (holdProgress * 100);

    return (
        <div className='bg-black/50 backdrop-blur-sm border border-white/10 p-6 rounded-2xl shadow-sm flex items-center justify-between'>
            <div>
                <div className='font-medium'>Hold to commit</div>
                <div className='text-xs text-gray-400'>Space = toggle hold · Ctrl+Enter = commit · Ctrl+Z = undo</div>
            </div>
            <div className='w-12 h-12 relative'>
                <svg viewBox='0 0 36 36' className='w-12 h-12'>
                    <path d='M18 2 a 16 16 0 1 1 0 32 a 16 16 0 1 1 0 -32' fill='none' stroke='#222' strokeWidth='4' />
                    <path d='M18 2 a 16 16 0 1 1 0 32 a 16 16 0 1 1 0 -32' fill='none' stroke='#7C3AED' strokeWidth='4' strokeDasharray='100' strokeDashoffset={`${offset}`} />
                </svg>
            </div>
        </div>
    );
}

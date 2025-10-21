'use client';
import { useStore } from '@/lib/store';

export function ModeSwitch() {
    const mode = useStore((s: any) => s.mode);
    const setMode = useStore((s: any) => s.setMode);
    return (
        <div className='inline-flex bg-white/10 border border-white/10 rounded-full p-1'>
            {(['letters', 'phrases', 'ensemble'] as const).map(m => (
                <button key={m} onClick={() => setMode(m)}
                    className={`px-3 py-1 rounded-full text-sm ${mode === m ? 'bg-white/10 border border-white/10 shadow' : ''}`}>{m}</button>
            ))}
        </div>
    );
}

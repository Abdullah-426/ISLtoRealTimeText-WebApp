'use client';
import { useStore } from '@/lib/store';

export function PhraseModeSwitch() {
    const phraseMode = useStore((s: any) => s.phraseMode);
    const setPhraseMode = useStore((s: any) => s.setPhraseMode);

    return (
        <div className='inline-flex bg-white/10 border border-white/10 rounded-full p-1'>
            {(['TCN', 'LSTM', 'Ensemble'] as const).map(m => (
                <button key={m} onClick={() => setPhraseMode(m)}
                    className={`px-3 py-1 rounded-full text-sm ${phraseMode === m ? 'bg-white/10 border border-white/10 shadow' : ''}`}>{m}</button>
            ))}
        </div>
    );
}

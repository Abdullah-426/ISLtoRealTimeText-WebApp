'use client';
import { useStore } from '@/lib/store';

export function PhraseModeSwitch() {
    const phraseV5Mode = useStore((s: any) => s.phraseV5Mode);
    const setPhraseV5Mode = useStore((s: any) => s.setPhraseV5Mode);

    return (
        <div className='inline-flex bg-white/10 border border-white/10 rounded-full p-1'>
            {(['TCN', 'LSTM', 'Ensemble'] as const).map(m => (
                <button key={m} onClick={() => setPhraseV5Mode(m)}
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${phraseV5Mode === m ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-300 hover:bg-white/5'}`}>{m}</button>
            ))}
        </div>
    );
}

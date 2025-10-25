'use client';
import { useStore } from '@/lib/store';

export default function PhraseV5ModeSwitch() {
    const phraseV5Mode = useStore((s: any) => s.phraseV5Mode);
    const setPhraseV5Mode = useStore((s: any) => s.setPhraseV5Mode);

    return (
        <div className='bg-[#0f0f19] border border-white/5 p-4 rounded-2xl shadow-sm'>
            <h3 className='text-white text-sm font-medium mb-3'>Phrase Model (V5)</h3>
            <div className='flex gap-2'>
                {(['TCN', 'LSTM', 'Ensemble'] as const).map((mode) => (
                    <button
                        key={mode}
                        onClick={() => setPhraseV5Mode(mode)}
                        className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${phraseV5Mode === mode
                                ? 'bg-blue-600 text-white'
                                : 'bg-white/5 text-gray-300 hover:bg-white/10'
                            }`}
                    >
                        {mode}
                    </button>
                ))}
            </div>
        </div>
    );
}

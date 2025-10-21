'use client';
import { useState } from 'react';
import { useStore } from '@/lib/store';

export function SettingsModal() {
    const [open, setOpen] = useState(false);
    const cfg = useStore((s: any) => s.config);
    const setCfg = useStore((s: any) => s.setConfig);

    return (
        <div className='bg-black/50 backdrop-blur-sm border border-white/10 p-6 rounded-2xl shadow-sm'>
            <div className='flex items-center justify-between'>
                <div className='font-medium'>Settings</div>
                <button className='px-3 py-1 rounded-full bg-white/10 border border-white/10' onClick={() => setOpen(v => !v)}>{open ? 'Close' : 'Open'}</button>
            </div>
            {open && (
                <div className='mt-3 space-y-2 text-sm'>
                    <label className='block'>Postprocess URL
                        <input className='mt-1 w-full border border-white/10 bg-black/20 rounded-lg p-2' value={cfg.postUrl}
                            onChange={e => setCfg({ postUrl: e.target.value })} />
                    </label>
                    <label className='block'>Phrase Infer Base URL
                        <input className='mt-1 w-full border border-white/10 bg-black/20 rounded-lg p-2' value={cfg.phraseBase}
                            onChange={e => setCfg({ phraseBase: e.target.value })} />
                    </label>
                    <label className='block'>LLM Provider
                        <select className='mt-1 w-full border border-white/10 bg-black/20 rounded-lg p-2' value={cfg.provider}
                            onChange={e => setCfg({ provider: e.target.value as any })}>
                            <option value='local'>local</option>
                            <option value='openai'>openai</option>
                            <option value='groq'>groq</option>
                        </select>
                    </label>
                    <p className='text-gray-500'>Front-end only sends committed tokens and compact feature windows. No video leaves your device.</p>
                </div>
            )}
        </div>
    );
}

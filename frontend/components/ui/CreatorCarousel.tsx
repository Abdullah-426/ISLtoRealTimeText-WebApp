'use client';
import { useEffect, useRef, useState } from 'react';

const CREATORS = ['Abdullah Ansari', 'Aryan Tayal', 'Pranav Bansal', 'Devyansh Kirar'];

export default function CreatorCarousel() {
    const [idx, setIdx] = useState(0);
    useEffect(() => {
        const t = setInterval(() => setIdx(i => (i + 1) % CREATORS.length), 2000);
        return () => clearInterval(t);
    }, []);
    return (
        <section className='bg-[#0f0f19] border border-white/5 p-6 rounded-2xl shadow-sm'>
            <div className='text-sm text-gray-400 mb-2'>Creators</div>
            <div className='flex gap-2 flex-wrap items-center'>
                {CREATORS.map((c, i) => (
                    <span key={c} className={`px-3 py-1 rounded-full border ${i === idx ? 'bg-white/10 border-white/10' : 'bg-black/20 border-white/10'}`}>{c}</span>
                ))}
            </div>
        </section>
    )
}

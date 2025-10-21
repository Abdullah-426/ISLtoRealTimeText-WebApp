const FEATURES = [
    { title: 'On-device landmarks', desc: 'Hand/Pose/Face via MediaPipe Tasks-Vision in the browser.' },
    { title: 'Letters (TF.js)', desc: '126-D hand features into MLP for A-Z/1-9/blank.' },
    { title: 'Phrases (LSTM/TCN)', desc: '1662-D holistic features, server inference with attention.' },
    { title: 'Smoothing & Commit', desc: 'EMA, hold-to-commit, presence gating, entropy guard.' },
    { title: 'LLM Postprocess', desc: 'Glue words & punctuation via OpenAI/Groq/local fallback.' },
    { title: 'Deployable', desc: 'Vercel (FE) + Railway (BE), no Docker required.' }
] as const;

export default function FeatureGrid() {
    return (
        <section className='grid md:grid-cols-3 gap-4'>
            {FEATURES.map(f => (
                <div key={f.title} className='bg-[#0f0f19] border border-white/5 p-6 rounded-2xl'>
                    <div className='font-medium'>{f.title}</div>
                    <div className='text-sm text-gray-400 mt-1'>{f.desc}</div>
                </div>
            ))}
        </section>
    )
}

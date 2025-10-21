export default function Hero() {
    return (
        <section className='relative overflow-hidden rounded-2xl border border-white/5 bg-gradient-to-br from-[#12121b] to-[#0b0b12] p-8'>
            <div className='max-w-5xl'>
                <h1 className='font-display text-4xl md:text-5xl'>ISL → Real-time Text</h1>
                <p className='text-gray-400 mt-3'>MediaPipe landmarks on-device · TF.js letters · Python LSTM/TCN phrases · LLM postprocess</p>
            </div>
            <div className='absolute -top-24 -right-24 w-96 h-96 bg-primary/20 blur-3xl rounded-full' />
            <div className='absolute -bottom-24 -left-24 w-96 h-96 bg-secondary/20 blur-3xl rounded-full' />
        </section>
    )
}

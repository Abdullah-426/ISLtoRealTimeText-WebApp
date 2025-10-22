'use client';
import { useStore } from '@/lib/store';

export default function ProcessedBar() {
    const processed = useStore((s: any) => s.processedText);
    const updateProcessedText = useStore((s: any) => s.updateProcessedText);
    const clearProcessed = useStore((s: any) => s.clearProcessed);
    const exportProcessed = useStore((s: any) => s.exportProcessed);

    return (
        <div className='bg-black/50 backdrop-blur-sm border border-white/10 p-6 rounded-2xl shadow-sm space-y-3 w-full'>
            <div className='text-sm text-gray-400'>Processed Output</div>
            <textarea
                className='min-h-[200px] w-full p-3 rounded-xl bg-black/30 border border-white/10 text-white placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent'
                value={processed}
                onChange={(e) => updateProcessedText(e.target.value)}
                placeholder="Processed text will appear here..."
            />
            <div className='flex gap-2'>
                <button className='px-4 py-2 rounded-full bg-white/10 border border-white/10' onClick={clearProcessed}>Clear</button>
                <button className='px-4 py-2 rounded-full bg-white/10 border border-white/10' onClick={exportProcessed}>Export .txt</button>
            </div>
        </div>
    );
}



'use client';
import { useStore } from '@/lib/store';

export function TypedBar() {
    const transcript = useStore((s: any) => s.transcript);
    const updateTranscript = useStore((s: any) => s.updateTranscript);
    const commitNow = useStore((s: any) => s.commitNow);
    const undo = useStore((s: any) => s.undo);
    const exportTxt = useStore((s: any) => s.exportTxt);

    const handleTranscriptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        updateTranscript(e.target.value);
    };

    return (
        <div className='bg-black/50 backdrop-blur-sm border border-white/10 p-6 rounded-2xl shadow-sm space-y-3'>
            <div className='text-sm text-gray-400'>Transcript</div>
            <textarea
                className='min-h-[120px] w-full p-3 rounded-xl bg-black/30 border border-white/10 text-white placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent'
                value={transcript}
                onChange={handleTranscriptChange}
                placeholder="Type your text here or use sign language recognition..."
            />
            <div className='flex gap-2'>
                <button className='bg-primary text-white px-4 py-2 rounded-full shadow hover:translate-y-0.5 transition' onClick={commitNow}>Commit</button>
                <button className='px-4 py-2 rounded-full bg-white/10 border border-white/10' onClick={undo}>Undo</button>
                <button className='px-4 py-2 rounded-full bg-white/10 border border-white/10' onClick={exportTxt}>Export .txt</button>
            </div>
        </div>
    );
}

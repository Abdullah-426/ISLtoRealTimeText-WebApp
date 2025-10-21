import { ModeSwitch } from '@/components/ModeSwitch';
import { TopK } from '@/components/TopK';
import { TypedBar } from '@/components/TypedBar';
import { CommitBar } from '@/components/CommitBar';
import WebcamPane from '@/components/WebcamPane';
import { SettingsModal } from '@/components/SettingsModal';
import { AnomalousMatterHero } from '@/components/anomalous-matter-hero';
import { CreatorsCarousel } from '@/components/creators-carousel';
import { GridFeatureCards } from '@/components/grid-feature-cards';
import { DottedSurface } from '@/components/dotted-surface';

export default function Page() {
    return (
        <main className='min-h-screen text-white relative z-10'>
            <AnomalousMatterHero />

            {/* Hero Text Section - At the very top */}
            <section className='py-16 md:py-24 text-center relative z-20'>
                <div className='max-w-5xl mx-auto px-4'>
                    <div className='bg-black/50 backdrop-blur-sm border border-white/10 rounded-2xl p-8 md:p-12 lg:p-16'>
                        <h1 className='text-base font-mono tracking-widest text-blue-400/80 uppercase mb-4'>
                            ISL RECOGNITION SYSTEM
                        </h1>
                        <p className='text-3xl md:text-5xl lg:text-6xl xl:text-7xl font-bold leading-tight text-white mb-6'>
                            Signs transform into words through digital alchemy.
                        </p>
                        <p className='mt-6 max-w-xl mx-auto text-base md:text-lg leading-relaxed text-gray-300/80'>
                            Advanced AI models decode the silent language of gestures, bridging the gap between movement and meaning in real-time.
                        </p>
                    </div>
                </div>
            </section>

            {/* Webcam and Controls Section - Horizontal layout */}
            <section className='container mx-auto px-4 py-8 relative z-20'>
                <div className='mx-auto max-w-6xl grid grid-cols-1 md:grid-cols-12 gap-6'>
                    <div className='md:col-span-7'>
                        <WebcamPane />
                        <div className='mt-4 space-y-4'>
                            <CommitBar />
                            <SettingsModal />
                        </div>
                    </div>
                    <div className='md:col-span-5 space-y-4'>
                        <div className='bg-black/50 backdrop-blur-sm border border-white/10 p-6 rounded-2xl shadow-sm'>
                            <div className='flex items-center justify-between'>
                                <h1 className='font-display text-2xl'>ISL → Text</h1>
                                <ModeSwitch />
                            </div>
                            <p className='text-sm text-gray-400 mt-2'>Client-side landmarks + TF.js letters. Phrases inferred server-side. Only committed tokens/feature windows are sent.</p>
                        </div>
                        <TopK />
                        <TypedBar />
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <div className="relative z-20">
                <GridFeatureCards />
            </div>

            {/* Creators Section */}
            <div className="py-12 bg-black/50 backdrop-blur-sm relative z-20">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-12">
                        <h2 className="text-4xl font-bold text-white mb-4">Meet the Creators</h2>
                        <p className="text-slate-300 text-lg">The brilliant minds behind this project</p>
                    </div>
                    <div className="py-6">
                        <CreatorsCarousel creators={[
                            {
                                id: '1',
                                name: 'Abdullah Ansari',
                                role: 'Lead Developer',
                                avatar: '',
                                description: '',
                                social: {
                                    github: 'https://github.com/abdullah-ansari',
                                    linkedin: 'https://linkedin.com/in/abdullah-ansari'
                                }
                            },
                            {
                                id: '2',
                                name: 'Aryan Tayal',
                                role: '',
                                avatar: '',
                                description: '',
                                social: {
                                    github: 'https://github.com/aryan-tayal',
                                    linkedin: 'https://linkedin.com/in/aryan-tayal'
                                }
                            },
                            {
                                id: '3',
                                name: 'Pranav Bansal',
                                role: '',
                                avatar: '',
                                description: '',
                                social: {
                                    github: 'https://github.com/pranav-bansal',
                                    linkedin: 'https://linkedin.com/in/pranav-bansal'
                                }
                            },
                            {
                                id: '4',
                                name: 'Devyansh Kirar',
                                role: '',
                                avatar: '',
                                description: '',
                                social: {
                                    github: 'https://github.com/devyansh-kirar',
                                    linkedin: 'https://linkedin.com/in/devyansh-kirar'
                                }
                            }
                        ]} />
                    </div>
                </div>
            </div>

            <DottedSurface />
        </main>
    );
}

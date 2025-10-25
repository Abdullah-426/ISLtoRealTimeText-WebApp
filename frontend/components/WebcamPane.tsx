'use client';
import { useEffect, useRef, useState } from 'react';
import { useStore } from '@/lib/store';
import { setupVision, runVisionFrame, resetVision } from '@/lib/vision';
import { motion } from 'framer-motion';
import { Camera, CameraOff, Pause, Play } from 'lucide-react';

export default function WebcamPane() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [ready, setReady] = useState(false);
    const [cameraOn, setCameraOn] = useState(false);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const onFeatures = useStore((s: any) => s.onFeatures);
    const onFeaturesV5 = useStore((s: any) => s.onFeaturesV5);
    const mode = useStore((s: any) => s.mode);
    const predictionsPaused = useStore((s: any) => s.predictionsPaused);
    const togglePredictionsPaused = useStore((s: any) => s.togglePredictionsPaused);

    const startProcessingLoop = () => {
        const loop = async () => {
            if (!videoRef.current || !canvasRef.current || !cameraOn) {
                return;
            }
            try {
                const frame = videoRef.current;
                const canvas = canvasRef.current;
                const res = await runVisionFrame(frame, canvas);

                // Detect hand presence from the vision results
                const leftHandPresent = res.leftHandPresent || false;
                const rightHandPresent = res.rightHandPresent || false;

                if (mode === 'phrases') {
                    // Use v5 phrase logic for phrases mode
                    onFeaturesV5({
                        vec1662: res.vec1662,
                        presenceRatio: res.presenceRatio,
                        leftHandPresent,
                        rightHandPresent
                    });
                } else {
                    // Use original logic for letters mode
                    onFeatures(res);
                }
            } catch (error) {
                console.error('Vision processing error:', error);
            }
            if (cameraOn) {
                animationFrameRef.current = requestAnimationFrame(loop);
            }
        };
        animationFrameRef.current = requestAnimationFrame(loop);
    };

    const stopProcessingLoop = () => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
    };

    const startCamera = async () => {
        try {
            const newStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
            setStream(newStream);
            if (videoRef.current) {
                videoRef.current.srcObject = newStream;
                await videoRef.current.play();
                setReady(true);
                setCameraOn(true);

                // Wait a bit for video to be ready
                await new Promise(resolve => setTimeout(resolve, 300));

                // Always reset and reload MediaPipe Tasks to ensure fresh initialization
                console.log('Reinitializing MediaPipe...');
                resetVision();
                await setupVision();
            }
        } catch (e) {
            console.error('Camera error', e);
        }
    };

    const stopCamera = () => {
        // Stop processing loop first
        stopProcessingLoop();

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        // Clear the canvas when camera is off
        if (canvasRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }
        setReady(false);
        setCameraOn(false);
    };

    const toggleCamera = () => {
        if (cameraOn) {
            stopCamera();
        } else {
            startCamera();
        }
    };

    useEffect(() => {
        // Cleanup on unmount
        return () => {
            stopProcessingLoop();
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [stream]);

    // Handle camera state changes
    useEffect(() => {
        if (cameraOn && ready) {
            // Camera is on and ready, start processing
            startProcessingLoop();
        } else {
            // Camera is off or not ready, stop processing
            stopProcessingLoop();
        }
    }, [cameraOn, ready]);

    return (
        <div className='bg-[#0f0f19] border border-white/5 p-4 rounded-2xl shadow-sm'>
            <div className='relative overflow-hidden rounded-2xl'>
                <video ref={videoRef} className='w-full aspect-video object-cover rounded-2xl scale-x-[-1]' muted playsInline />
                <canvas ref={canvasRef} className='absolute inset-0 w-full h-full scale-x-[-1]'></canvas>
                {!ready && (
                    <div className='absolute inset-0 grid place-items-center text-gray-500'>Allow camera access…</div>
                )}
                {cameraOn && ready && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className='absolute bottom-2 right-2 bg-black/50 text-white text-xs rounded-full px-2 py-1'>
                        {predictionsPaused ? 'Paused' : 'Live'}
                    </motion.div>
                )}
                {/* Camera Toggle Button */}
                <button
                    onClick={toggleCamera}
                    className='absolute top-2 right-2 bg-black/50 backdrop-blur-sm border border-white/10 text-white p-2 rounded-full hover:bg-black/70 transition-colors'
                    aria-label={cameraOn ? 'Turn off camera' : 'Turn on camera'}
                >
                    {cameraOn ? <CameraOff className='w-4 h-4' /> : <Camera className='w-4 h-4' />}
                </button>
                {/* Predictions Pause/Resume Button */}
                {cameraOn && ready && (
                    <button
                        onClick={togglePredictionsPaused}
                        className='absolute bottom-2 left-2 bg-black/50 backdrop-blur-sm border border-white/10 text-white p-2 rounded-full hover:bg-black/70 transition-colors'
                        aria-label={predictionsPaused ? 'Resume predictions' : 'Pause predictions'}
                    >
                        {predictionsPaused ? <Play className='w-4 h-4' /> : <Pause className='w-4 h-4' />}
                    </button>
                )}
            </div>
        </div>
    );
}

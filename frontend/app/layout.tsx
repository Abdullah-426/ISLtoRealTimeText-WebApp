import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'ISL → Text',
    description: 'Real-time ISL to Text with MediaPipe + TF.js + FastAPI',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}

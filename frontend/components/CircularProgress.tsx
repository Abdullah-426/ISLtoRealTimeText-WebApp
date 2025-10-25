'use client';
import { useEffect, useRef } from 'react';

interface CircularProgressProps {
    progress: number; // 0 to 1
    size?: number;
    strokeWidth?: number;
    color?: string;
    backgroundColor?: string;
    children?: React.ReactNode;
}

export function CircularProgress({
    progress,
    size = 120,
    strokeWidth = 8,
    color = '#10b981', // green-500
    backgroundColor = '#374151', // gray-700
    children
}: CircularProgressProps) {
    const circleRef = useRef<SVGCircleElement>(null);
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDasharray = circumference;

    useEffect(() => {
        if (circleRef.current) {
            const strokeDashoffset = circumference - (progress * circumference);
            circleRef.current.style.strokeDashoffset = strokeDashoffset.toString();
        }
    }, [progress, circumference]);

    return (
        <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
            <svg
                width={size}
                height={size}
                className="transform -rotate-90"
            >
                {/* Background circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={backgroundColor}
                    strokeWidth={strokeWidth}
                    fill="none"
                />
                {/* Progress circle - smooth CSS transition */}
                <circle
                    ref={circleRef}
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={color}
                    strokeWidth={strokeWidth}
                    fill="none"
                    strokeLinecap="round"
                    strokeDasharray={strokeDasharray}
                    strokeDashoffset={circumference}
                    style={{
                        transition: 'stroke-dashoffset 0.05s linear',
                        transformOrigin: 'center'
                    }}
                />
            </svg>
            {/* Content inside the circle */}
            {children && (
                <div className="absolute inset-0 flex items-center justify-center">
                    {children}
                </div>
            )}
        </div>
    );
}

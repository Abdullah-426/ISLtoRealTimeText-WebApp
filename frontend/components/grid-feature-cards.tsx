import { cn } from '@/lib/utils';
import React from 'react';
import { GlowingEffect } from '@/components/ui/glowing-effect';

type FeatureType = {
	title: string;
	icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
	description: string;
};

type FeatureCardPorps = React.ComponentProps<'div'> & {
	feature: FeatureType;
};

export function FeatureCard({ feature, className, ...props }: FeatureCardPorps) {
	const p = genRandomPattern();

	return (
		<div className={cn('relative overflow-hidden p-6', className)} {...props}>
			<GlowingEffect
				disabled={false}
				glow={true}
				blur={35}
				spread={120}
				proximity={120}
				inactiveZone={0.1}
				movementDuration={1.2}
				borderWidth={6}
				variant="default"
				className="rounded-xl"
			/>
			<div className="pointer-events-none absolute top-0 left-1/2 -mt-2 -ml-20 h-full w-full [mask-image:linear-gradient(white,transparent)]">
				<div className="from-foreground/5 to-foreground/1 absolute inset-0 bg-gradient-to-r [mask-image:radial-gradient(farthest-side_at_top,white,transparent)] opacity-100">
					<GridPattern
						width={20}
						height={20}
						x="-12"
						y="4"
						squares={p}
						className="fill-foreground/5 stroke-foreground/25 absolute inset-0 h-full w-full mix-blend-overlay"
					/>
				</div>
			</div>
			<feature.icon className="text-foreground/75 size-6" strokeWidth={1} aria-hidden />
			<h3 className="mt-10 text-sm md:text-base">{feature.title}</h3>
			<p className="text-muted-foreground relative z-20 mt-2 text-xs font-light">{feature.description}</p>
		</div>
	);
}

function GridPattern({
	width,
	height,
	x,
	y,
	squares,
	...props
}: React.ComponentProps<'svg'> & { width: number; height: number; x: string; y: string; squares?: number[][] }) {
	const patternId = React.useId();

	return (
		<svg aria-hidden="true" {...props}>
			<defs>
				<pattern id={patternId} width={width} height={height} patternUnits="userSpaceOnUse" x={x} y={y}>
					<path d={`M.5 ${height}V.5H${width}`} fill="none" />
				</pattern>
			</defs>
			<rect width="100%" height="100%" strokeWidth={0} fill={`url(#${patternId})`} />
			{squares && (
				<svg x={x} y={y} className="overflow-visible">
					{squares.map(([x, y], index) => (
						<rect strokeWidth="0" key={index} width={width + 1} height={height + 1} x={x * width} y={y * height} />
					))}
				</svg>
			)}
		</svg>
	);
}

function genRandomPattern(length?: number): number[][] {
	length = length ?? 5;
	return Array.from({ length }, () => [
		Math.floor(Math.random() * 4) + 7, // random x between 7 and 10
		Math.floor(Math.random() * 6) + 1, // random y between 1 and 6
	]);
}

export function GridFeatureCards() {
	const features = [
		{
			title: "Real-time Recognition",
			icon: (props: React.SVGProps<SVGSVGElement>) => (
				<svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
					<path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
				</svg>
			),
			description: "Advanced AI models process sign language in real-time with high accuracy."
		},
		{
			title: "Multi-Modal Support",
			icon: (props: React.SVGProps<SVGSVGElement>) => (
				<svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
					<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
					<circle cx="9" cy="7" r="4" />
					<path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
				</svg>
			),
			description: "Supports both individual letters and complete phrases for comprehensive communication."
		},
		{
			title: "Client-Side Processing",
			icon: (props: React.SVGProps<SVGSVGElement>) => (
				<svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
					<rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
					<line x1="8" y1="21" x2="16" y2="21" />
					<line x1="12" y1="17" x2="12" y2="21" />
				</svg>
			),
			description: "Landmark detection runs locally in your browser for privacy and speed."
		},
		{
			title: "Ensemble Learning",
			icon: (props: React.SVGProps<SVGSVGElement>) => (
				<svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
					<path d="M9 12l2 2 4-4" />
					<path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3" />
					<path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3" />
					<path d="M12 3c0 1-1 3-3 3s-3-2-3-3 1-3 3-3 3 2 3 3" />
					<path d="M12 21c0-1 1-3 3-3s3 2 3 3-1 3-3 3-3-2-3-3" />
				</svg>
			),
			description: "Combines multiple AI models for the most accurate predictions possible."
		},
		{
			title: "Natural Language Processing",
			icon: (props: React.SVGProps<SVGSVGElement>) => (
				<svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
					<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
				</svg>
			),
			description: "LLM-powered post-processing adds punctuation and natural language flow."
		},
		{
			title: "Accessibility First",
			icon: (props: React.SVGProps<SVGSVGElement>) => (
				<svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
					<path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
					<circle cx="12" cy="12" r="3" />
				</svg>
			),
			description: "Designed to break down communication barriers and make technology accessible to everyone."
		}
	];

	return (
		<section className="py-16 bg-black/50 backdrop-blur-sm">
			<div className="container mx-auto px-4">
				<div className="text-center mb-12">
					<h2 className="text-4xl font-bold text-white mb-4">Powerful Features</h2>
					<p className="text-slate-300 text-lg">Everything you need for seamless sign language recognition</p>
				</div>
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
					{features.map((feature, index) => (
						<FeatureCard key={index} feature={feature} className="bg-black/50 backdrop-blur-sm border border-white/10 rounded-xl" />
					))}
				</div>
			</div>
		</section>
	);
}

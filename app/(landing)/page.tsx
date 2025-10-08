import Link from 'next/link';
import Image from 'next/image';

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-neutral-950 to-neutral-900 text-neutral-100">
      <div className="container mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="flex flex-col items-center text-center mb-16">
          <Image
            src="/logo.png"
            alt="SpideyPlot Logo"
            width={120}
            height={120}
            className="mb-8"
          />
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-orange-400 to-orange-600 bg-clip-text text-transparent">
            SpideyPlot
          </h1>
          <p className="text-xl text-neutral-400 max-w-2xl mb-8">
            Electrochemical impedance spectroscopy (EIS) simulation platform
            for retinal pigment epithelium (RPE) research
          </p>
          <Link
            href="/simulator"
            className="px-8 py-4 bg-orange-600 hover:bg-orange-700 rounded-lg text-lg font-semibold transition-colors shadow-lg hover:shadow-xl"
          >
            Launch Simulator
          </Link>
        </div>
      </div>
    </main>
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function FeatureCard({ title, description, icon }: { title: string; description: string; icon: string }) {
  return (
    <div className="bg-neutral-800/50 border border-neutral-700 rounded-lg p-6 hover:border-orange-600 transition-colors">
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-neutral-400">{description}</p>
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function QuickLink({ href, title, description }: { href: string; title: string; description: string }) {
  return (
    <Link
      href={href}
      className="block bg-neutral-800/30 border border-neutral-700 rounded-lg p-6 hover:bg-neutral-800/50 hover:border-orange-600 transition-all"
    >
      <h4 className="text-lg font-semibold mb-2 text-orange-400">{title}</h4>
      <p className="text-sm text-neutral-400">{description}</p>
    </Link>
  );
}

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import 'katex/dist/katex.min.css';
import { AuthProvider } from './components/auth/AuthProvider';

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SpideyPlot - NEI Visualization Tool",
  description: "Advanced visualization tool for bioimpedance circuit analysis and parameter exploration",
  icons: {
    icon: '/spiderweb.png',
    shortcut: '/spiderweb.png',
    apple: '/spiderweb.png',
  },
  openGraph: {
    title: "SpideyPlot - NEI Visualization Tool",
    description: "Advanced electrochemical impedance spectroscopy (EIS) simulation for retinal pigment epithelium research with 3D visualization and parameter exploration",
    siteName: "SpideyPlot",
    url: "https://nei-viz-project.vercel.app",
    type: "website",
    images: [
      {
        url: "/spiderweb.png",
        width: 512,
        height: 512,
        alt: "SpideyPlot - EIS Visualization Tool",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "SpideyPlot - NEI Visualization Tool",
    description: "Advanced EIS simulation and visualization for bioimpedance circuit analysis",
    images: ["/spiderweb.png"],
  },
  metadataBase: new URL("https://nei-viz-project.vercel.app"),
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}

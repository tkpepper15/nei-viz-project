import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import '@radix-ui/themes/styles.css';
import 'katex/dist/katex.min.css';
import { Theme } from '@radix-ui/themes';
import { AuthProvider } from './components/auth/AuthProvider';

const ibmPlexSans = IBM_Plex_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
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
        className={`${ibmPlexSans.variable} ${ibmPlexMono.variable} antialiased bg-background text-foreground`}
      >
        <Theme appearance="dark" accentColor="grass" grayColor="sage" radius="medium" scaling="95%">
          <AuthProvider>
            {children}
          </AuthProvider>
        </Theme>
      </body>
    </html>
  );
}

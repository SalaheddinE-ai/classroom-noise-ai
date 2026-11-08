import type { Metadata } from "next";
import "./globals.css";
import Shell from "@/components/Shell";

export const metadata: Metadata = { title: "Classroom Noise AI", description: "Classroom noise monitor dashboard" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* apply the saved theme before first paint (no flash) */}
        <script dangerouslySetInnerHTML={{ __html: "try{var t=localStorage.getItem('cnai-theme');if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)}catch(e){}" }} />
      </head>
      <body><Shell>{children}</Shell></body>
    </html>
  );
}

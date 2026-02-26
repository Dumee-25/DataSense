import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "DataSense — AI Data Analysis",
  description: "Upload your CSV. Get expert-level insights instantly.",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

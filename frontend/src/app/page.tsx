"use client"
import { useState, useCallback } from "react"
import { useRouter } from "next/navigation"
import NavBar from "@/components/NavBar"
import { uploadCSV } from "@/lib/datasense-api"

export default function UploadPage() {
  const router = useRouter()
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy]         = useState(false)
  const [error, setError]       = useState("")

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.endsWith(".csv")) { setError("Only CSV files are supported."); return }
    setError(""); setBusy(true)
    try {
      const { job_id } = await uploadCSV(file)
      router.push(`/analyzing/${job_id}`)
    } catch (e: any) { setError(e.message || "Upload failed."); setBusy(false) }
  }, [router])

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      <NavBar />

      {/* subtle grid */}
      <div className="fixed inset-0 pointer-events-none" style={{
        backgroundImage: "linear-gradient(rgba(0,217,181,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,217,181,0.03) 1px,transparent 1px)",
        backgroundSize: "64px 64px",
      }} />

      <main className="relative flex flex-col items-center justify-center min-h-screen px-6 pt-14 gap-10">

        {/* hero text */}
        <div className="text-center max-w-xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium mb-5 fade-up"
            style={{ background: "var(--teal-dim)", color: "var(--teal)", border: "1px solid rgba(0,217,181,0.2)" }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "var(--teal)" }} />
            AI-powered · runs locally
          </div>
          <h1 className="text-5xl font-bold leading-tight mb-3 fade-up delay-1" style={{ fontFamily: "Syne, sans-serif" }}>
            Your data,{" "}
            <span style={{ color: "var(--teal)" }}>expertly analyzed</span>
          </h1>
          <p className="text-base fade-up delay-2" style={{ color: "var(--dim)" }}>
            Upload any CSV and get a full quality audit, statistical findings,
            and model recommendations in plain English — in seconds.
          </p>
        </div>

        {/* drop zone */}
        <div className="w-full max-w-lg fade-up delay-3"
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }}>

          <label htmlFor="csv-input"
            className="flex flex-col items-center justify-center w-full rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 p-14"
            style={{
              borderColor: dragging ? "var(--teal)" : "var(--border)",
              background: dragging ? "var(--teal-dim)" : "var(--surface)",
              boxShadow: dragging ? "0 0 40px rgba(0,217,181,0.12)" : "none",
            }}>
            {busy ? (
              <div className="flex flex-col items-center gap-3">
                <div className="w-10 h-10 rounded-full border-2 animate-spin"
                  style={{ borderColor: "var(--teal)", borderTopColor: "transparent" }} />
                <span className="text-sm" style={{ color: "var(--teal)" }}>Uploading…</span>
              </div>
            ) : (
              <>
                <div className="w-14 h-14 rounded-xl flex items-center justify-center mb-4 transition-transform hover:scale-110"
                  style={{ background: "var(--teal-dim)" }}>
                  <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" style={{ color: "var(--teal)" }}>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <p className="font-semibold text-base mb-1" style={{ fontFamily: "Syne, sans-serif" }}>
                  {dragging ? "Drop it here" : "Drop your CSV here"}
                </p>
                <p className="text-sm" style={{ color: "var(--muted)" }}>
                  or <span style={{ color: "var(--teal)" }}>click to browse</span>
                </p>
                <p className="text-xs mt-2" style={{ color: "var(--muted)" }}>CSV only · max 50 MB</p>
              </>
            )}
            <input id="csv-input" type="file" accept=".csv" className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} disabled={busy} />
          </label>
        </div>

        {error && <p className="text-sm px-4 py-2 rounded-lg fade-up" style={{ color: "var(--critical)", background: "rgba(239,68,68,0.08)" }}>{error}</p>}

        {/* feature pills */}
        <div className="flex flex-wrap justify-center gap-2 fade-up delay-4">
          {["Data quality audit","Statistical tests","Model recommendation","PDF report","LLM insights"].map(f => (
            <span key={f} className="px-3 py-1.5 rounded-full text-xs"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--dim)" }}>
              ✓ {f}
            </span>
          ))}
        </div>
      </main>
    </div>
  )
}

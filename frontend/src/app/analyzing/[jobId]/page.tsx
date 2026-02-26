"use client"
import { useEffect, useState } from "react"
import { useRouter, useParams } from "next/navigation"
import NavBar from "@/components/NavBar"
import { fetchJobStatus, cancelAnalysis, JobStatus } from "@/lib/datasense-api"

const PIPELINE_STEPS = [
  { label: "Validating file",            at: 5  },
  { label: "Loading dataset",            at: 10 },
  { label: "Analyzing structure",        at: 30 },
  { label: "Running statistical tests",  at: 55 },
  { label: "Building model suggestions", at: 75 },
  { label: "Generating insights",        at: 90 },
  { label: "Saving results",            at: 98 },
]

export default function ProgressPage() {
  const router    = useRouter()
  const { jobId } = useParams<{ jobId: string }>()
  const [job, setJob]           = useState<JobStatus | null>(null)
  const [cancelling, setCancel] = useState(false)

  useEffect(() => {
    if (!jobId) return
    const poll = async () => {
      try {
        const s = await fetchJobStatus(jobId)
        setJob(s)
        if (s.status === "completed") { setTimeout(() => router.push(`/results/${jobId}`), 500); return }
        if (s.status !== "failed" && s.status !== "cancelled") setTimeout(poll, 1500)
      } catch { setTimeout(poll, 3000) }
    }
    poll()
  }, [jobId, router])

  const handleCancel = async () => { setCancel(true); await cancelAnalysis(jobId); router.push("/") }

  const pct        = job?.progress ?? 0
  const isFailed   = job?.status === "failed"
  const isDone     = job?.status === "completed"
  const activeStep = [...PIPELINE_STEPS].reverse().find(s => pct >= s.at) ?? PIPELINE_STEPS[0]

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      <NavBar />
      <div className="fixed inset-0 pointer-events-none" style={{
        background: "radial-gradient(circle at 50% 45%, rgba(0,217,181,0.04) 0%, transparent 65%)",
      }} />

      <main className="relative flex flex-col items-center justify-center min-h-screen px-6 pt-14">
        <div className="w-full max-w-md">

          {job?.filename && (
            <p className="text-center text-sm mb-8" style={{ color: "var(--muted)" }}>
              Analyzing <span className="font-mono" style={{ color: "var(--dim)" }}>{job.filename}</span>
            </p>
          )}

          {isFailed ? (
            <div className="text-center fade-up">
              <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
                style={{ background: "rgba(239,68,68,0.1)", border: "2px solid var(--critical)" }}>
                <span className="text-2xl">✕</span>
              </div>
              <h2 className="text-xl font-bold mb-2"
                style={{ fontFamily: "Syne, sans-serif", color: "var(--critical)" }}>
                Analysis failed
              </h2>
              <p className="text-sm mb-5" style={{ color: "var(--muted)" }}>
                {job?.error || "An unexpected error occurred."}
              </p>
              <button onClick={() => router.push("/")}
                className="px-5 py-2.5 rounded-xl text-sm font-semibold"
                style={{ background: "var(--surface-2)", color: "var(--text)" }}>
                ← Try again
              </button>
            </div>

          ) : isDone ? (
            <div className="text-center fade-up">
              <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3"
                style={{ background: "rgba(0,217,181,0.1)", border: "2px solid var(--teal)" }}>
                <span className="text-2xl">✓</span>
              </div>
              <p className="text-sm" style={{ color: "var(--teal)" }}>Complete — redirecting…</p>
            </div>

          ) : (
            <>
              {/* pulsing orb */}
              <div className="flex justify-center mb-8">
                <div className="relative w-20 h-20">
                  <div className="absolute inset-0 rounded-full animate-ping opacity-10"
                    style={{ background: "var(--teal)" }} />
                  <div className="absolute inset-2 rounded-full"
                    style={{ background: "var(--teal-dim)", border: "2px solid var(--teal)" }} />
                  <div className="absolute inset-0 flex items-center justify-center font-mono text-sm font-bold"
                    style={{ color: "var(--teal)" }}>
                    {pct}%
                  </div>
                </div>
              </div>

              {/* progress bar */}
              <div className="w-full h-1 rounded-full mb-2 overflow-hidden" style={{ background: "var(--surface-2)" }}>
                <div className="h-full rounded-full shimmer transition-all duration-700" style={{ width: `${pct}%` }} />
              </div>
              <p className="text-center text-sm mb-7" style={{ color: "var(--dim)" }}>
                {job?.progress_message || "Starting…"}
              </p>

              {/* step list */}
              <div className="space-y-1.5">
                {PIPELINE_STEPS.map((step, i) => {
                  const done   = pct > step.at
                  const active = activeStep.label === step.label
                  return (
                    <div key={i}
                      className="flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all"
                      style={{
                        background: active ? "var(--teal-dim)" : "transparent",
                        border: `1px solid ${active ? "rgba(0,217,181,0.25)" : "transparent"}`,
                      }}>
                      <div className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                        style={{
                          background: done || active ? "var(--teal)" : "var(--surface-2)",
                          color: done || active ? "#070D1A" : "var(--muted)",
                        }}>
                        {done ? "✓" : i + 1}
                      </div>
                      <span className="text-sm"
                        style={{ color: done || active ? "var(--text)" : "var(--muted)" }}>
                        {step.label}
                      </span>
                      {active && (
                        <span className="ml-auto text-xs animate-pulse" style={{ color: "var(--teal)" }}>
                          running
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>

              <div className="flex justify-center mt-7">
                <button onClick={handleCancel} disabled={cancelling}
                  className="px-4 py-2 rounded-lg text-sm disabled:opacity-40"
                  style={{ color: "var(--muted)", border: "1px solid var(--border)" }}>
                  {cancelling ? "Cancelling…" : "Cancel"}
                </button>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}

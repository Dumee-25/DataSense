"use client"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import NavBar from "@/components/NavBar"
import { fetchHistory, removeJob, HistoryJob } from "@/lib/datasense-api"

const STATUS_DOT: Record<string, string> = {
  completed: "var(--success)", processing: "var(--teal)",
  queued: "var(--medium)", failed: "var(--critical)", cancelled: "var(--muted)",
}

function ago(d: string) {
  const s = Math.floor((Date.now() - new Date(d).getTime()) / 1000)
  if (s < 60) return "just now"
  if (s < 3600) return `${Math.floor(s/60)}m ago`
  if (s < 86400) return `${Math.floor(s/3600)}h ago`
  return `${Math.floor(s/86400)}d ago`
}

export default function HistoryPage() {
  const router = useRouter()
  const [jobs, setJobs]       = useState<HistoryJob[]>([])
  const [loading, setLoading] = useState(true)
  const [removing, setRemoving] = useState<string|null>(null)

  useEffect(() => { fetchHistory().then(r => { setJobs(r.jobs); setLoading(false) }) }, [])

  const del = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm("Delete this analysis and all its data?")) return
    setRemoving(id)
    await removeJob(id)
    setJobs(prev => prev.filter(j => j.job_id !== id))
    setRemoving(null)
  }

  const open = (job: HistoryJob) => {
    if (job.status === "completed") router.push(`/results/${job.job_id}`)
    else if (["queued","processing"].includes(job.status)) router.push(`/analyzing/${job.job_id}`)
  }

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      <NavBar />
      <div className="fixed inset-0 pointer-events-none" style={{
        backgroundImage: "linear-gradient(rgba(0,217,181,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(0,217,181,0.02) 1px,transparent 1px)",
        backgroundSize: "64px 64px",
      }} />

      <main className="relative max-w-3xl mx-auto px-5 pt-24 pb-16">
        <div className="flex items-center justify-between mb-7 fade-up">
          <div>
            <h1 className="text-2xl font-bold" style={{ fontFamily: "Syne, sans-serif" }}>History</h1>
            <p className="text-sm mt-0.5" style={{ color: "var(--muted)" }}>
              {jobs.length} {jobs.length === 1 ? "analysis" : "analyses"} this session
            </p>
          </div>
          <button onClick={() => router.push("/")} className="px-4 py-2 rounded-xl text-sm font-semibold"
            style={{ background: "var(--teal)", color: "#070D1A", fontFamily: "Syne, sans-serif" }}>
            + New analysis
          </button>
        </div>

        {loading ? (
          <div className="flex justify-center py-20">
            <div className="w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: "var(--teal)", borderTopColor: "transparent" }} />
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-20 fade-up">
            <p className="text-5xl mb-3">📂</p>
            <p className="font-semibold mb-1" style={{ fontFamily: "Syne, sans-serif" }}>No analyses yet</p>
            <p className="text-sm mb-5" style={{ color: "var(--muted)" }}>Upload a CSV to get started.</p>
            <button onClick={() => router.push("/")} className="px-5 py-2.5 rounded-xl font-semibold text-sm"
              style={{ background: "var(--teal)", color: "#070D1A" }}>Upload your first file</button>
          </div>
        ) : (
          <div className="space-y-2.5 fade-up">
            {jobs.map(job => (
              <div key={job.job_id} onClick={() => open(job)}
                className="group flex items-center gap-4 p-4 rounded-2xl border cursor-pointer transition-all"
                style={{ background: "var(--surface)", borderColor: "var(--border)" }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = "var(--border-hover)")}
                onMouseLeave={e => (e.currentTarget.style.borderColor = "var(--border)")}>

                <div className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ background: STATUS_DOT[job.status] || "var(--muted)" }} />

                <div className="flex-1 min-w-0">
                  <p className="font-semibold truncate" style={{ fontFamily: "Syne, sans-serif" }}>{job.filename}</p>
                  <div className="flex items-center gap-3 mt-0.5 text-xs" style={{ color: "var(--muted)" }}>
                    <span>{ago(job.created_at)}</span>
                    {job.processing_time_seconds && <span>· {job.processing_time_seconds.toFixed(1)}s</span>}
                    <span className="capitalize font-medium" style={{ color: STATUS_DOT[job.status] }}>{job.status}</span>
                  </div>
                </div>

                {job.metadata && (
                  <div className="hidden md:flex items-center gap-4 text-xs flex-shrink-0">
                    <div className="text-center">
                      <p className="font-mono font-semibold" style={{ color: "var(--teal)" }}>{job.metadata.rows?.toLocaleString()}</p>
                      <p style={{ color: "var(--muted)" }}>rows</p>
                    </div>
                    <div className="text-center">
                      <p className="font-mono" style={{ color: "var(--dim)" }}>{job.metadata.columns}</p>
                      <p style={{ color: "var(--muted)" }}>cols</p>
                    </div>
                    {job.metadata.critical_issues > 0 && (
                      <div className="text-center">
                        <p className="font-mono font-semibold" style={{ color: "var(--critical)" }}>{job.metadata.critical_issues}</p>
                        <p style={{ color: "var(--muted)" }}>critical</p>
                      </div>
                    )}
                    <span className="px-2 py-0.5 rounded text-xs" style={{ background: "var(--surface-2)", color: "var(--muted)" }}>
                      {job.metadata.primary_model}
                    </span>
                  </div>
                )}

                {["processing","queued"].includes(job.status) && (
                  <div className="w-16 h-1 rounded-full overflow-hidden flex-shrink-0" style={{ background: "var(--surface-2)" }}>
                    <div className="h-full rounded-full shimmer" style={{ width: `${job.progress}%` }} />
                  </div>
                )}

                <button onClick={e => del(job.job_id, e)} disabled={removing === job.job_id}
                  className="opacity-0 group-hover:opacity-100 w-7 h-7 flex items-center justify-center rounded-lg transition-all flex-shrink-0 text-sm"
                  style={{ color: "var(--muted)" }}
                  onMouseEnter={e => (e.currentTarget.style.color = "var(--critical)")}
                  onMouseLeave={e => (e.currentTarget.style.color = "var(--muted)")}>
                  {removing === job.job_id ? "…" : "✕"}
                </button>
              </div>
            ))}
          </div>
        )}

        {jobs.length > 0 && !loading && (
          <p className="text-center text-xs mt-8" style={{ color: "var(--muted)" }}>
            Sign in to keep your history across devices and browser sessions.
          </p>
        )}
      </main>
    </div>
  )
}

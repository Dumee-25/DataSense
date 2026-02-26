"use client"
import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import NavBar from "@/components/NavBar"
import { fetchResults, pdfDownloadUrl, FullResults, Insight } from "@/lib/datasense-api"

type Tab = "summary" | "findings" | "relations" | "model" | "columns"

const SEV = {
  critical: { color: "var(--critical)", bg: "rgba(239,68,68,0.08)" },
  high:     { color: "var(--high)",     bg: "rgba(249,115,22,0.08)" },
  medium:   { color: "var(--medium)",   bg: "rgba(234,179,8,0.08)"  },
}

function Badge({ sev }: { sev: string }) {
  const s = SEV[sev as keyof typeof SEV] || { color: "var(--muted)", bg: "var(--surface-2)" }
  return (
    <span className="px-2 py-0.5 rounded text-xs font-bold flex-shrink-0"
      style={{ background: s.bg, color: s.color }}>
      {sev.toUpperCase()}
    </span>
  )
}

function FindingCard({ item }: { item: Insight }) {
  const [open, setOpen] = useState(false)
  const col = Array.isArray(item.column) ? item.column.join(", ") : item.column
  const borderColor = (SEV[item.severity as keyof typeof SEV] || { color: "var(--border)" }).color
  const isCritical  = item.severity === "critical"

  return (
    <div className="rounded-xl border overflow-hidden"
      style={{ borderColor: `${borderColor}33`, background: "var(--surface)" }}>

      {/* ── always-visible header ── */}
      <button className="w-full text-left px-4 py-3.5 flex items-start gap-3"
        onClick={() => setOpen(!open)}>
        <Badge sev={item.severity} />
        <div className="flex-1">
          <p className="text-sm font-medium" style={{ color: "var(--text)" }}>{item.headline}</p>
          {col && (
            <p className="text-xs font-mono mt-0.5" style={{ color: "var(--muted)" }}>{col}</p>
          )}
        </div>
        <span className="text-xs mt-1 flex-shrink-0"
          style={{ color: "var(--muted)", transition: "transform 0.15s", display: "inline-block", transform: open ? "rotate(180deg)" : "none" }}>
          ▾
        </span>
      </button>

      {/* ── expanded body ── */}
      {open && (
        <div className="border-t" style={{ borderColor: "rgba(255,255,255,0.05)" }}>

          {/* What it means */}
          {item.what_it_means && (
            <div className="px-4 pt-3 pb-2">
              <p className="text-xs font-semibold mb-1" style={{ color: "var(--muted)" }}>WHAT IT MEANS</p>
              <p className="text-sm leading-relaxed" style={{ color: "var(--dim)" }}>{item.what_it_means}</p>
            </div>
          )}

          {/* Business impact — NEW */}
          {item.business_impact && (
            <div className="px-4 py-2">
              <div className="p-3 rounded-lg" style={{
                background: isCritical ? "rgba(239,68,68,0.06)" : "rgba(249,115,22,0.06)"
              }}>
                <p className="text-xs font-semibold mb-1" style={{ color: borderColor }}>BUSINESS IMPACT</p>
                <p className="text-sm leading-relaxed" style={{ color: "var(--dim)" }}>{item.business_impact}</p>
              </div>
            </div>
          )}

          {/* Action */}
          {item.what_to_do && (
            <div className="px-4 py-2">
              <div className="p-3 rounded-lg" style={{ background: "var(--surface-2)" }}>
                <p className="text-xs font-semibold mb-1" style={{ color: "var(--teal)" }}>ACTION</p>
                <p className="text-sm whitespace-pre-wrap" style={{ color: "var(--text)" }}>{item.what_to_do}</p>
              </div>
            </div>
          )}

          {/* Deep dive — NEW, only on critical issues */}
          {item.deep_dive && (
            <div className="px-4 py-2 pb-4">
              <div className="p-3 rounded-lg border"
                style={{ background: "rgba(139,92,246,0.04)", borderColor: "rgba(139,92,246,0.2)" }}>
                <div className="flex items-center gap-1.5 mb-1">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="rgb(167,139,250)"
                    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/>
                  </svg>
                  <p className="text-xs font-semibold" style={{ color: "rgb(167,139,250)" }}>AI DEEP DIVE</p>
                </div>
                <p className="text-sm leading-relaxed" style={{ color: "var(--dim)" }}>{item.deep_dive}</p>
              </div>
            </div>
          )}

          {!item.deep_dive && !item.what_to_do && <div className="pb-3" />}
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, warn }: { label: string; value: string | number; warn?: boolean }) {
  return (
    <div className="p-4 rounded-xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <p className="text-xs mb-1" style={{ color: "var(--muted)" }}>{label}</p>
      <p className="text-2xl font-bold font-mono" style={{ color: warn ? "var(--critical)" : "var(--teal)" }}>
        {value}
      </p>
    </div>
  )
}

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const router    = useRouter()
  const [data, setData]       = useState<FullResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab]         = useState<Tab>("summary")

  useEffect(() => {
    fetchResults(jobId).then(r => { setData(r); setLoading(false) }).catch(() => router.push("/"))
  }, [jobId, router])

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--bg)" }}>
      <div className="w-8 h-8 rounded-full border-2 animate-spin"
        style={{ borderColor: "var(--teal)", borderTopColor: "transparent" }} />
    </div>
  )
  if (!data) return null

  const ins         = data.results.insights
  const basic       = data.results.structural_analysis?.basic_info || data.results.dataset_info || {}
  const guide       = ins?.model_guidance || {}
  const bd          = ins?.severity_breakdown || {}
  const allFindings = [...(ins?.critical_insights||[]),...(ins?.high_priority_insights||[]),...(ins?.medium_priority_insights||[])]
  const profiles    = data.results.structural_analysis?.column_profiles || []
  const colRels     = ins?.column_relationships || []
  const imbalance   = ins?.class_imbalance_guidance || null
  const totalIssues = (bd.critical||0)+(bd.high||0)+(bd.medium||0)
  const llmEnhanced = ins?.llm_enhanced ?? false

  const TABS: { id: Tab; label: string }[] = [
    { id: "summary",   label: "Summary" },
    { id: "findings",  label: `Findings (${totalIssues})` },
    { id: "relations", label: `Relations (${colRels.length})` },
    { id: "model",     label: "Model" },
    { id: "columns",   label: `Columns (${basic.columns||0})` },
  ]

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      <NavBar />
      <div className="max-w-4xl mx-auto px-5 pt-20 pb-16">

        {/* header */}
        <div className="flex items-start justify-between mb-7 fade-up">
          <div>
            <p className="text-xs font-mono mb-1" style={{ color: "var(--muted)" }}>Analysis complete</p>
            <h1 className="text-2xl font-bold" style={{ fontFamily: "Syne, sans-serif" }}>{data.filename}</h1>
            <p className="text-sm mt-0.5 flex items-center gap-2" style={{ color: "var(--muted)" }}>
              in {data.processing_time_seconds?.toFixed(1)}s
              {llmEnhanced && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
                  style={{ background: "rgba(139,92,246,0.12)", color: "rgb(167,139,250)" }}>
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/>
                  </svg>
                  AI-Enhanced
                  {ins?.llm_provider && <span className="opacity-70">· {ins.llm_provider}</span>}
                </span>
              )}
              {ins?.persona && ins.persona !== 'executive' && (
                <span className="px-2 py-0.5 rounded-full text-xs"
                  style={{ background: "rgba(0,217,181,0.1)", color: "var(--teal)" }}>
                  {ins.persona.replace('_', ' ')}
                </span>
              )}
            </p>
          </div>
          <div className="flex gap-2">
            <button onClick={() => router.push("/")} className="px-4 py-2 rounded-xl text-sm"
              style={{ background: "var(--surface)", color: "var(--dim)", border: "1px solid var(--border)" }}>
              ↑ New
            </button>
            <a href={pdfDownloadUrl(jobId)} target="_blank" rel="noopener noreferrer"
              className="px-4 py-2 rounded-xl text-sm font-semibold"
              style={{ background: "var(--teal)", color: "#070D1A", fontFamily: "Syne, sans-serif" }}>
              ↓ PDF
            </a>
          </div>
        </div>

        {/* stat cards */}
        <div className="grid grid-cols-4 gap-3 mb-6 fade-up delay-1">
          <Stat label="Rows"    value={(basic.rows||0).toLocaleString()} />
          <Stat label="Columns" value={basic.columns||0} />
          <Stat label="Missing" value={`${basic.missing_percentage||0}%`} warn={(basic.missing_percentage||0)>15} />
          <Stat label="Issues"  value={ins?.total_insights||0} warn={(bd.critical||0)>0} />
        </div>

        {/* executive summary */}
        {ins?.executive_summary && (
          <div className="p-5 rounded-2xl mb-6 fade-up delay-2"
            style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2 mb-2">
              <p className="text-xs font-semibold" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>
                EXECUTIVE SUMMARY
              </p>
              {llmEnhanced && (
                <span className="text-xs px-1.5 py-0.5 rounded"
                  style={{ background: "rgba(139,92,246,0.1)", color: "rgb(167,139,250)" }}>AI</span>
              )}
              {ins?.generation_time_seconds != null && (
                <span className="text-xs" style={{ color: "var(--muted)" }}>
                  Insights generated in {ins.generation_time_seconds.toFixed(1)}s
                </span>
              )}
            </div>
            <p className="text-sm leading-relaxed" style={{ color: "var(--dim)" }}>{ins.executive_summary}</p>
          </div>
        )}

        {/* tab bar */}
        <div className="flex gap-1 mb-5 p-1 rounded-xl fade-up delay-3" style={{ background: "var(--surface)" }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className="flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all"
              style={{
                background: tab === t.id ? "var(--surface-2)" : "transparent",
                color: tab === t.id ? "var(--text)" : "var(--muted)",
              }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* ══ SUMMARY ══ */}
        {tab === "summary" && (
          <div className="space-y-4 fade-up">
            {ins?.quick_wins?.length > 0 && (
              <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <p className="text-xs font-semibold mb-3" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>
                  QUICK WINS — DO THESE FIRST
                </p>
                <ol className="space-y-2">
                  {ins.quick_wins.map((w: string, i: number) => (
                    <li key={i} className="flex gap-3 text-sm">
                      <span className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                        style={{ background: "var(--teal)", color: "#070D1A" }}>{i+1}</span>
                      <span style={{ color: "var(--dim)" }}>{w}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
            <div className="grid grid-cols-3 gap-3">
              {[{l:"Critical",n:bd.critical,c:"var(--critical)"},{l:"High",n:bd.high,c:"var(--high)"},{l:"Medium",n:bd.medium,c:"var(--medium)"}]
                .map(({l,n,c}) => (
                  <div key={l} className="p-4 rounded-xl text-center" style={{ background: "var(--surface)", border: `1px solid ${c}33` }}>
                    <p className="text-3xl font-bold font-mono" style={{ color: c }}>{n||0}</p>
                    <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>{l} issues</p>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* ══ FINDINGS ══ */}
        {tab === "findings" && (
          <div className="space-y-2.5 fade-up">
            {allFindings.length === 0 ? (
              <div className="text-center py-14" style={{ color: "var(--muted)" }}>
                <p className="text-4xl mb-2">✓</p>
                <p>No significant issues — data looks clean.</p>
              </div>
            ) : allFindings.map((f,i) => <FindingCard key={i} item={f} />)}
          </div>
        )}

        {/* ══ RELATIONS ══ */}
        {tab === "relations" && (
          <div className="space-y-4 fade-up">
            {imbalance && (
              <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid rgba(239,68,68,0.25)" }}>
                <p className="text-xs font-semibold mb-2" style={{ color: "var(--critical)", fontFamily: "Syne, sans-serif" }}>
                  CLASS IMBALANCE — "{imbalance.target_column}"
                </p>
                <p className="text-sm leading-relaxed mb-4" style={{ color: "var(--dim)" }}>{imbalance.why_it_matters}</p>
                <p className="text-xs font-semibold mb-2" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>HOW TO FIX IT</p>
                <div className="space-y-2.5 mb-4">
                  {imbalance.techniques?.map((t: any, i: number) => (
                    <div key={i} className="p-3 rounded-lg" style={{ background: "var(--surface-2)" }}>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>{t.name}</span>
                        <span className="px-2 py-0.5 rounded text-xs" style={{
                          background: t.difficulty === 'easy' ? 'rgba(0,217,181,0.1)' : 'rgba(234,179,8,0.1)',
                          color: t.difficulty === 'easy' ? 'var(--teal)' : 'var(--medium)',
                        }}>{t.difficulty}</span>
                      </div>
                      <p className="text-sm mb-2" style={{ color: "var(--dim)" }}>{t.description}</p>
                      <code className="text-xs px-2 py-1 rounded block"
                        style={{ background: "rgba(0,0,0,0.3)", color: "var(--muted)", fontFamily: "monospace" }}>
                        {t.code_hint}
                      </code>
                    </div>
                  ))}
                </div>
                <div className="p-3 rounded-lg" style={{ background: "rgba(239,68,68,0.06)" }}>
                  <p className="text-xs font-semibold mb-1" style={{ color: "var(--critical)" }}>METRICS</p>
                  <p className="text-sm mb-2" style={{ color: "var(--dim)" }}>{imbalance.metric_explanation}</p>
                  <div className="flex flex-wrap gap-4">
                    <div>
                      <span className="text-xs" style={{ color: "var(--critical)" }}>✗ Don't use: </span>
                      <span className="text-xs" style={{ color: "var(--muted)" }}>{imbalance.wrong_metrics?.join(", ")}</span>
                    </div>
                    <div>
                      <span className="text-xs" style={{ color: "var(--teal)" }}>✓ Use: </span>
                      <span className="text-xs" style={{ color: "var(--muted)" }}>{imbalance.right_metrics?.join(", ")}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {colRels.length === 0 && !imbalance ? (
              <div className="text-center py-14" style={{ color: "var(--muted)" }}>
                <p className="text-4xl mb-2">~</p>
                <p>No notable column relationships found.</p>
              </div>
            ) : colRels.map((r: any, i: number) => {
              const relColor = r.severity==='critical'?'var(--critical)':r.severity==='high'?'var(--high)':r.severity==='medium'?'var(--medium)':'var(--muted)'
              return (
                <div key={i} className="p-4 rounded-xl border" style={{ background: "var(--surface)", borderColor: `${relColor}33` }}>
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <span className="text-sm font-mono font-semibold" style={{ color: "var(--teal)" }}>{r.col_a}</span>
                    <span className="text-xs" style={{ color: "var(--muted)" }}>↔</span>
                    <span className="text-sm font-mono font-semibold" style={{ color: "var(--teal)" }}>{r.col_b}</span>
                    {r.correlation != null && (
                      <span className="ml-auto text-xs font-mono px-2 py-0.5 rounded"
                        style={{ color: relColor, background: `${relColor}15` }}>
                        {r.correlation > 0 ? '+' : ''}{r.correlation.toFixed(2)} · {r.strength}
                      </span>
                    )}
                    {r.split_by && (
                      <span className="ml-auto text-xs px-2 py-0.5 rounded"
                        style={{ color: 'var(--critical)', background: 'rgba(239,68,68,0.08)' }}>
                        reverses by "{r.split_by}"
                      </span>
                    )}
                  </div>
                  {r.correlation != null && (
                    <div className="w-full h-1.5 rounded-full mb-3" style={{ background: "var(--surface-2)" }}>
                      <div className="h-full rounded-full" style={{
                        width: `${Math.abs(r.correlation)*100}%`,
                        background: r.correlation>0?'var(--teal)':'var(--critical)', opacity: 0.7,
                      }} />
                    </div>
                  )}
                  <p className="text-sm leading-relaxed mb-2" style={{ color: "var(--dim)" }}>{r.explanation}</p>
                  <div className="p-2.5 rounded-lg" style={{ background: "var(--surface-2)" }}>
                    <p className="text-xs font-semibold mb-0.5" style={{ color: "var(--teal)" }}>ACTION</p>
                    <p className="text-sm" style={{ color: "var(--text)" }}>{r.action}</p>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* ══ MODEL ══ */}
        {tab === "model" && (
          <div className="space-y-4 fade-up">
            <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="flex items-start justify-between mb-3">
                <div>
                  <p className="text-xs font-semibold mb-1" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>RECOMMENDED MODEL</p>
                  <h2 className="text-2xl font-bold" style={{ fontFamily: "Syne, sans-serif" }}>{guide.recommended_model}</h2>
                  <p className="text-sm mt-0.5" style={{ color: "var(--muted)" }}>{guide.task_type} · {guide.confidence_label}</p>
                </div>
                <span className="px-3 py-1 rounded-full text-xs font-semibold" style={{ background: "var(--teal-dim)", color: "var(--teal)" }}>
                  {Math.round((guide.confidence_score||0)*100)}% confidence
                </span>
              </div>
              {guide.why_this_model && <p className="text-sm mb-3 leading-relaxed" style={{ color: "var(--dim)" }}>{guide.why_this_model}</p>}
              {guide.key_reasons?.map((r: string, i: number) => (
                <p key={i} className="text-sm flex gap-2 mb-1" style={{ color: "var(--dim)" }}>
                  <span style={{ color: "var(--teal)" }}>→</span>{r}
                </p>
              ))}
            </div>
            {guide.alternatives?.length > 0 && (
              <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <p className="text-xs font-semibold mb-3" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>ALTERNATIVES</p>
                {guide.alternatives.map((a: any, i: number) => (
                  <div key={i} className="flex gap-3 text-sm mb-2">
                    <span className="font-semibold" style={{ color: "var(--text)" }}>{a.model}</span>
                    <span style={{ color: "var(--muted)" }}>— {a.why}</span>
                  </div>
                ))}
              </div>
            )}
            {guide.before_you_train?.length > 0 && (
              <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <p className="text-xs font-semibold mb-3" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>BEFORE YOU TRAIN</p>
                <ol className="space-y-2">
                  {guide.before_you_train.map((s: string, i: number) => (
                    <li key={i} className="flex gap-3 text-sm">
                      <span className="font-mono text-xs mt-0.5 flex-shrink-0" style={{ color: "var(--muted)" }}>{i+1}.</span>
                      <span style={{ color: "var(--dim)" }}>{s}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
            {guide.how_to_validate && (
              <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <p className="text-xs font-semibold mb-2" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>HOW TO VALIDATE</p>
                <p className="text-sm" style={{ color: "var(--dim)" }}>{guide.how_to_validate}</p>
              </div>
            )}
            {guide.how_to_measure_success?.length > 0 && (
              <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <p className="text-xs font-semibold mb-3" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>SUCCESS METRICS</p>
                {guide.how_to_measure_success.map((m: string, i: number) => (
                  <p key={i} className="text-sm mb-1" style={{ color: "var(--dim)" }}>• {m}</p>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ══ COLUMNS ══ */}
        {tab === "columns" && (
          <div className="fade-up rounded-2xl overflow-hidden border" style={{ borderColor: "var(--border)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr style={{ background: "var(--surface-2)" }}>
                  {["Column","Type","Missing","Unique","Notes"].map(h => (
                    <th key={h} className="text-left px-4 py-3 text-xs font-semibold"
                      style={{ color: "var(--muted)", fontFamily: "Syne, sans-serif" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {profiles.map((p: any, i: number) => {
                  const notes: string[] = []
                  if (p.missing_pct >= 50) notes.push("high missingness")
                  if (p.skewness && Math.abs(p.skewness) > 2) notes.push(`skewed (${p.skewness?.toFixed(1)})`)
                  if (p.disguised_missing > 0) notes.push(`${p.disguised_missing} disguised nulls`)
                  if (p.inf_count > 0) notes.push(`${p.inf_count} ∞`)
                  return (
                    <tr key={i} style={{ background: i%2===0?"var(--surface)":"var(--surface-2)", borderTop: "1px solid var(--border)" }}>
                      <td className="px-4 py-3 font-mono font-medium" style={{ color: "var(--text)" }}>{p.name}</td>
                      <td className="px-4 py-3" style={{ color: "var(--muted)" }}>{p.kind?.replace(/_/g," ")}</td>
                      <td className="px-4 py-3 font-mono" style={{ color: p.missing_pct>20?"var(--critical)":"var(--dim)" }}>{p.missing_pct}%</td>
                      <td className="px-4 py-3 font-mono" style={{ color: "var(--dim)" }}>{p.unique_count?.toLocaleString()}</td>
                      <td className="px-4 py-3 text-xs" style={{ color: "var(--muted)" }}>{notes.join(" · ")||"—"}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
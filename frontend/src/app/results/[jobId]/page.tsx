"use client"
import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import NavBar from "@/components/NavBar"
import { fetchResults, fetchCharts, pdfDownloadUrl, FullResults, Insight, ChartData, DomainContext } from "@/lib/datasense-api"

type Tab = "summary" | "findings" | "charts" | "relations" | "model" | "columns"

const SEV = {
  critical: { color: "var(--critical)", bg: "rgba(239,68,68,0.08)" },
  high:     { color: "var(--high)",     bg: "rgba(249,115,22,0.08)" },
  medium:   { color: "var(--medium)",   bg: "rgba(234,179,8,0.08)"  },
}

const PRIO = {
  "must-fix":      { label: "MUST FIX", color: "#EF4444", bg: "rgba(239,68,68,0.12)" },
  "should-fix":    { label: "SHOULD FIX", color: "#F97316", bg: "rgba(249,115,22,0.10)" },
  "nice-to-have":  { label: "NICE TO HAVE", color: "#EAB308", bg: "rgba(234,179,8,0.10)" },
  "informational": { label: "INFO", color: "#64748B", bg: "rgba(100,116,139,0.10)" },
}

function PrioBadge({ priority }: { priority?: string }) {
  if (!priority || !(priority in PRIO)) return null
  const p = PRIO[priority as keyof typeof PRIO]
  return (
    <span className="px-2 py-0.5 rounded text-[10px] font-bold tracking-wide flex-shrink-0"
      style={{ background: p.bg, color: p.color }}>
      {p.label}
    </span>
  )
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

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="px-2 py-0.5 rounded text-xs transition-colors"
      style={{ background: copied ? "rgba(0,217,181,0.15)" : "var(--surface-2)", color: copied ? "var(--teal)" : "var(--muted)" }}
      onClick={(e) => {
        e.stopPropagation()
        navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      }}>
      {copied ? "✓ copied" : "copy"}
    </button>
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
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-sm font-medium" style={{ color: "var(--text)" }}>{item.headline}</p>
            {item.aggregated && item.count && (
              <span className="px-1.5 py-0.5 rounded text-xs font-mono"
                style={{ background: "rgba(0,217,181,0.1)", color: "var(--teal)" }}>
                ×{item.count}
              </span>
            )}
            <PrioBadge priority={item.action_priority} />
          </div>
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

          {/* Model context note — shows relevance to recommended model (Fix 4) */}
          {item.model_context_note && (
            <div className="px-4 py-2">
              <div className="p-3 rounded-lg" style={{ background: "rgba(0,217,181,0.06)" }}>
                <p className="text-xs font-semibold mb-1" style={{ color: "var(--teal)" }}>MODEL CONTEXT</p>
                <p className="text-sm leading-relaxed" style={{ color: "var(--dim)" }}>{item.model_context_note}</p>
              </div>
            </div>
          )}

          {/* Aggregated sub-items — expandable list of affected columns/pairs */}
          {item.aggregated && item.affected_columns && item.affected_columns.length > 0 && (
            <div className="px-4 py-2">
              <p className="text-xs font-semibold mb-1.5" style={{ color: "var(--muted)" }}>AFFECTED COLUMNS</p>
              <div className="flex flex-wrap gap-1.5">
                {item.affected_columns.map((c: string) => (
                  <span key={c} className="px-2 py-0.5 rounded text-xs font-mono"
                    style={{ background: "var(--surface-2)", color: "var(--dim)" }}>{c}</span>
                ))}
              </div>
            </div>
          )}

          {item.aggregated && item.pairs && item.pairs.length > 0 && (
            <div className="px-4 py-2">
              <p className="text-xs font-semibold mb-1.5" style={{ color: "var(--muted)" }}>CORRELATED PAIRS</p>
              <div className="space-y-1">
                {item.pairs.slice(0, 8).map((p: any, i: number) => (
                  <div key={i} className="flex items-center gap-2 text-xs font-mono"
                    style={{ color: "var(--dim)" }}>
                    <span>{p.var1}</span>
                    <span style={{ color: "var(--muted)" }}>↔</span>
                    <span>{p.var2}</span>
                    <span className="ml-auto" style={{ color: "var(--teal)" }}>{p.correlation > 0 ? '+' : ''}{p.correlation.toFixed(3)}</span>
                  </div>
                ))}
                {item.pairs.length > 8 && (
                  <p className="text-xs" style={{ color: "var(--muted)" }}>
                    …and {item.pairs.length - 8} more pairs
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Business impact */}
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

          {/* Action with copy button */}
          {item.what_to_do && (
            <div className="px-4 py-2">
              <div className="p-3 rounded-lg" style={{ background: "var(--surface-2)" }}>
                <div className="flex items-center justify-between mb-1">
                  <p className="text-xs font-semibold" style={{ color: "var(--teal)" }}>ACTION</p>
                  <CopyButton text={item.what_to_do} />
                </div>
                <p className="text-sm whitespace-pre-wrap" style={{ color: "var(--text)" }}>{item.what_to_do}</p>
              </div>
            </div>
          )}

          {/* Deep dive */}
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
  const [sevFilter, setSevFilter] = useState<Record<string,boolean>>({ critical: true, high: true, medium: true })
  const [corrThreshold, setCorrThreshold] = useState(0.0)
  const [charts, setCharts]   = useState<Record<string, string | null>>({})
  const [chartsLoading, setChartsLoading] = useState(false)

  useEffect(() => {
    fetchResults(jobId).then(r => { setData(r); setLoading(false) }).catch(() => router.push("/"))
  }, [jobId, router])

  // Lazy-load charts when the tab is first selected
  useEffect(() => {
    if (tab === "charts" && Object.keys(charts).length === 0 && !chartsLoading) {
      setChartsLoading(true)
      fetchCharts(jobId).then(r => setCharts(r.charts || {})).finally(() => setChartsLoading(false))
    }
  }, [tab, jobId, charts, chartsLoading])

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
  const dataStory   = ins?.data_story || ""
  const domainCtx   = ins?.domain_context || null

  const TABS: { id: Tab; label: string }[] = [
    { id: "summary",   label: "Summary" },
    { id: "findings",  label: `Findings (${totalIssues})` },
    { id: "charts",    label: "Charts" },
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

            {/* Data Story — AI narrative */}
            {dataStory && (
              <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid rgba(139,92,246,0.2)" }}>
                <div className="flex items-center gap-2 mb-2">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgb(167,139,250)"
                    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/>
                  </svg>
                  <p className="text-xs font-semibold" style={{ color: "rgb(167,139,250)", fontFamily: "Syne, sans-serif" }}>
                    DATA STORY
                  </p>
                  <span className="text-xs px-1.5 py-0.5 rounded"
                    style={{ background: "rgba(139,92,246,0.1)", color: "rgb(167,139,250)" }}>AI</span>
                </div>
                <p className="text-sm leading-relaxed" style={{ color: "var(--dim)" }}>{dataStory}</p>
              </div>
            )}

            {/* Domain Context — LLM-inferred */}
            {domainCtx && domainCtx.domain !== "unknown" && domainCtx.confidence >= 0.4 && (
              <div className="p-5 rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <div className="flex items-center gap-2 mb-3">
                  <p className="text-xs font-semibold" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>
                    DETECTED CONTEXT
                  </p>
                  <span className="text-xs px-1.5 py-0.5 rounded"
                    style={{ background: "rgba(139,92,246,0.1)", color: "rgb(167,139,250)" }}>AI</span>
                  <span className="text-xs font-mono" style={{ color: "var(--muted)" }}>
                    {Math.round(domainCtx.confidence * 100)}% confidence
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm mb-3">
                  <div>
                    <span className="text-xs" style={{ color: "var(--muted)" }}>Domain: </span>
                    <span style={{ color: "var(--text)" }}>{domainCtx.domain}</span>
                  </div>
                  {domainCtx.purpose && (
                    <div>
                      <span className="text-xs" style={{ color: "var(--muted)" }}>Purpose: </span>
                      <span style={{ color: "var(--text)" }}>{domainCtx.purpose}</span>
                    </div>
                  )}
                  {domainCtx.target_meaning && (
                    <div className="col-span-2">
                      <span className="text-xs" style={{ color: "var(--muted)" }}>Target: </span>
                      <span style={{ color: "var(--text)" }}>{domainCtx.target_meaning}</span>
                    </div>
                  )}
                </div>
                {domainCtx.key_risks?.length > 0 && (
                  <div className="mb-2">
                    <p className="text-xs mb-1" style={{ color: "var(--muted)" }}>Domain risks:</p>
                    <div className="flex flex-wrap gap-1.5">
                      {domainCtx.key_risks.map((r, i) => (
                        <span key={i} className="px-2 py-0.5 rounded text-xs"
                          style={{ background: "rgba(239,68,68,0.08)", color: "var(--high)" }}>{r}</span>
                      ))}
                    </div>
                  </div>
                )}
                {domainCtx.leakage_suspects?.length > 0 && (
                  <div className="mb-2">
                    <p className="text-xs mb-1" style={{ color: "var(--muted)" }}>Leakage suspects:</p>
                    <div className="flex flex-wrap gap-1.5">
                      {domainCtx.leakage_suspects.map((c, i) => (
                        <span key={i} className="px-2 py-0.5 rounded text-xs font-mono"
                          style={{ background: "rgba(239,68,68,0.08)", color: "var(--critical)" }}>{c}</span>
                      ))}
                    </div>
                  </div>
                )}
                {Object.keys(domainCtx.column_meanings || {}).length > 0 && (
                  <details className="mt-2">
                    <summary className="text-xs cursor-pointer" style={{ color: "var(--muted)" }}>
                      Column meanings ({Object.keys(domainCtx.column_meanings).length} columns)
                    </summary>
                    <div className="mt-2 space-y-1 max-h-40 overflow-y-auto">
                      {Object.entries(domainCtx.column_meanings).map(([col, meaning]) => (
                        <div key={col} className="flex gap-2 text-xs">
                          <span className="font-mono flex-shrink-0" style={{ color: "var(--teal)" }}>{col}</span>
                          <span style={{ color: "var(--dim)" }}>{meaning}</span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}

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
        {tab === "findings" && (() => {
          const filtered = allFindings.filter(f => sevFilter[f.severity] !== false)
          return (
            <div className="space-y-2.5 fade-up">
              {/* severity toggle bar */}
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs" style={{ color: "var(--muted)" }}>Filter:</span>
                {(["critical","high","medium"] as const).map(s => {
                  const active = sevFilter[s] !== false
                  const c = SEV[s].color
                  return (
                    <button key={s}
                      className="px-2.5 py-1 rounded-full text-xs font-semibold transition-all"
                      style={{
                        background: active ? `${c}22` : "var(--surface-2)",
                        color: active ? c : "var(--muted)",
                        border: `1px solid ${active ? `${c}44` : "transparent"}`,
                      }}
                      onClick={() => setSevFilter(p => ({ ...p, [s]: !p[s] }))}>
                      {s.charAt(0).toUpperCase()+s.slice(1)}
                    </button>
                  )
                })}
                <span className="ml-auto text-xs font-mono" style={{ color: "var(--muted)" }}>
                  {filtered.length}/{allFindings.length}
                </span>
              </div>
              {filtered.length === 0 ? (
                <div className="text-center py-14" style={{ color: "var(--muted)" }}>
                  <p className="text-4xl mb-2">✓</p>
                  <p>{allFindings.length === 0 ? "No significant issues — data looks clean." : "All findings filtered out. Toggle a severity level above."}</p>
                </div>
              ) : filtered.map((f,i) => <FindingCard key={i} item={f} />)}
            </div>
          )
        })()}

        {/* ══ CHARTS ══ */}
        {tab === "charts" && (
          <div className="space-y-4 fade-up">
            {chartsLoading ? (
              <div className="text-center py-14">
                <div className="w-8 h-8 rounded-full border-2 animate-spin mx-auto mb-3"
                  style={{ borderColor: "var(--teal)", borderTopColor: "transparent" }} />
                <p className="text-sm" style={{ color: "var(--muted)" }}>Generating charts…</p>
              </div>
            ) : Object.values(charts).every(v => v === null) || Object.keys(charts).length === 0 ? (
              <div className="text-center py-14" style={{ color: "var(--muted)" }}>
                <p className="text-4xl mb-2">📊</p>
                <p>No charts available. Charts require matplotlib on the server.</p>
              </div>
            ) : (
              <>
                {[
                  { key: "data_health_radar", label: "Data Health Radar" },
                  { key: "missing_values",    label: "Missing Values" },
                  { key: "correlation",       label: "Correlations" },
                  { key: "target_distribution", label: "Target Distribution" },
                  { key: "outliers",          label: "Outlier Severity" },
                  { key: "skewness",          label: "Distribution Skewness" },
                ].map(({ key, label }) => {
                  const b64 = charts[key]
                  if (!b64) return null
                  return (
                    <div key={key} className="rounded-2xl overflow-hidden"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <div className="px-5 py-3 flex items-center justify-between"
                        style={{ borderBottom: "1px solid var(--border)" }}>
                        <p className="text-xs font-semibold" style={{ color: "var(--teal)", fontFamily: "Syne, sans-serif" }}>
                          {label.toUpperCase()}
                        </p>
                        <a href={`data:image/png;base64,${b64}`} download={`${key}.png`}
                          className="px-2 py-0.5 rounded text-xs transition-colors"
                          style={{ background: "var(--surface-2)", color: "var(--muted)" }}>
                          ↓ PNG
                        </a>
                      </div>
                      <div className="p-4 flex justify-center" style={{ background: "#FAFAFA" }}>
                        <img src={`data:image/png;base64,${b64}`} alt={label}
                          className="max-w-full h-auto rounded" style={{ maxHeight: 420 }} />
                      </div>
                    </div>
                  )
                })}
              </>
            )}
          </div>
        )}

        {/* ══ RELATIONS ══ */}
        {tab === "relations" && (
          <div className="space-y-4 fade-up">
            {/* correlation threshold slider */}
            {colRels.length > 0 && (
              <div className="flex items-center gap-3 p-3 rounded-xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <span className="text-xs whitespace-nowrap" style={{ color: "var(--muted)" }}>Min |r|:</span>
                <input type="range" min={0} max={0.99} step={0.01}
                  value={corrThreshold}
                  onChange={e => setCorrThreshold(parseFloat(e.target.value))}
                  className="flex-1 accent-teal-400 h-1.5" />
                <span className="text-xs font-mono w-10 text-right" style={{ color: "var(--teal)" }}>
                  {corrThreshold.toFixed(2)}
                </span>
              </div>
            )}
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
                      <div className="relative group">
                        <code className="text-xs px-2 py-1 rounded block pr-14"
                          style={{ background: "rgba(0,0,0,0.3)", color: "var(--muted)", fontFamily: "monospace" }}>
                          {t.code_hint}
                        </code>
                        <div className="absolute top-0.5 right-1">
                          <CopyButton text={t.code_hint} />
                        </div>
                      </div>
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
            ) : colRels.filter((r: any) => r.correlation == null || Math.abs(r.correlation) >= corrThreshold).length === 0 ? (
              <div className="text-center py-14" style={{ color: "var(--muted)" }}>
                <p className="text-4xl mb-2">~</p>
                <p>No relationships above |r| ≥ {corrThreshold.toFixed(2)}. Lower the threshold.</p>
              </div>
            ) : colRels.filter((r: any) => r.correlation == null || Math.abs(r.correlation) >= corrThreshold).map((r: any, i: number) => {
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
                    <div className="flex items-center justify-between mb-0.5">
                      <p className="text-xs font-semibold" style={{ color: "var(--teal)" }}>ACTION</p>
                      <CopyButton text={r.action} />
                    </div>
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
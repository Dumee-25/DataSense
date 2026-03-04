const BASE = "/api"

export interface JobStatus {
  job_id: string
  filename: string
  status: "queued" | "processing" | "completed" | "failed" | "cancelled"
  progress: number
  progress_message: string
  created_at: string
  completed_at?: string
  processing_time_seconds?: number
  error?: string
  preview?: { rows: number; columns: number; critical_issues: number; high_issues: number; primary_model: string }
}

export interface HistoryJob {
  job_id: string
  filename: string
  status: string
  progress: number
  created_at: string
  completed_at?: string
  processing_time_seconds?: number
  metadata?: {
    rows: number; columns: number; missing_pct: number
    critical_issues: number; high_issues: number; medium_issues: number; primary_model: string
  }
}

export interface Insight {
  type: string; severity: string; headline: string
  what_it_means: string; what_to_do: string; column?: string | string[]
  business_impact?: string; deep_dive?: string
  aggregated?: boolean; count?: number
  model_context_note?: string
  action_priority?: "must-fix" | "should-fix" | "nice-to-have" | "informational" | ""
  pairs?: { var1: string; var2: string; correlation: number }[]
  affected_columns?: string[]
}

export interface ColumnRelationship {
  col_a: string; col_b: string; split_by?: string
  correlation: number | null; strength: string
  direction: string; explanation: string
  action: string; severity: string
}

export interface ImbalanceTechnique {
  name: string; difficulty: string
  description: string; code_hint: string
}

export interface ClassImbalanceGuidance {
  target_column: string; majority_pct: number
  n_classes: number; imbalance_ratio: number | null
  why_it_matters: string
  techniques: ImbalanceTechnique[]
  wrong_metrics: string[]; right_metrics: string[]
  metric_explanation: string
}

export interface ModelGuide {
  recommended_model: string; task_type: string; confidence_label: string
  confidence_score: number; why_this_model: string; key_reasons: string[]
  alternatives: { model: string; why: string }[]
  before_you_train: string[]; how_to_validate: string; how_to_measure_success: string[]
}

export interface DomainContext {
  domain: string; purpose: string; target_meaning: string
  key_risks: string[]; leakage_suspects: string[]; metadata_cols: string[]
  column_meanings: Record<string, string>; confidence: number
}

export interface ChartData {
  job_id: string
  charts: Record<string, string | null>  // chart_name → base64 PNG or null
}

export interface FullResults {
  job_id: string; filename: string
  completed_at: string; processing_time_seconds: number
  results: {
    dataset_info: any; structural_analysis: any
    statistical_analysis: any; model_recommendations: any
    insights: {
      executive_summary: string
      data_story?: string
      domain_context?: DomainContext | null
      user_context_provided?: boolean
      critical_insights: Insight[]; high_priority_insights: Insight[]; medium_priority_insights: Insight[]
      column_relationships: ColumnRelationship[]
      class_imbalance_guidance: ClassImbalanceGuidance | null
      model_guidance: ModelGuide; quick_wins: string[]
      total_insights: number; severity_breakdown: { critical: number; high: number; medium: number }
      llm_enhanced?: boolean; persona?: string; llm_provider?: string | null
      generation_time_seconds?: number; metrics?: Record<string, any> | null
    }
  }
}

export interface AuthUser { id: string; email: string; name?: string; role: string; created_at: string }

async function call(path: string, opts?: RequestInit) {
  const res = await fetch(`${BASE}${path}`, { credentials: "include", ...opts })
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `Request failed (${res.status})`) }
  return res
}

export const uploadCSV = async (file: File, context?: string, targetColumn?: string) => {
  const form = new FormData(); form.append("file", file)
  if (context) form.append("context", context)
  if (targetColumn) form.append("target_column", targetColumn)
  return (await call("/analyze", { method: "POST", body: form })).json()
}

export const fetchJobStatus  = (id: string): Promise<JobStatus>   => call(`/status/${id}`).then(r => r.json())
export const fetchResults    = (id: string): Promise<FullResults>  => call(`/results/${id}`).then(r => r.json())
export const cancelAnalysis  = (id: string) => call(`/cancel/${id}`, { method: "DELETE" })
export const fetchHistory    = (): Promise<{ total: number; jobs: HistoryJob[] }> => call("/jobs").then(r => r.json()).catch(() => ({ total: 0, jobs: [] }))
export const removeJob       = (id: string) => call(`/jobs/${id}`, { method: "DELETE" })
export const pdfDownloadUrl  = (id: string) => `${BASE}/results/${id}/export/pdf`
export const fetchCharts    = (id: string): Promise<ChartData> => call(`/results/${id}/charts`).then(r => r.json()).catch(() => ({ job_id: id, charts: {} }))

export const fetchCurrentUser = (): Promise<{ user: AuthUser | null; authenticated: boolean }> =>
  call("/auth/me").then(r => r.json()).catch(() => ({ user: null, authenticated: false }))

export const signIn = (email: string, password: string) =>
  call("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) }).then(r => r.json())

export const signUp = (email: string, password: string, name?: string) =>
  call("/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password, name }) }).then(r => r.json())

export const signOut = () => call("/auth/logout", { method: "POST" })

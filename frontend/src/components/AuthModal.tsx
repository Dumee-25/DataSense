"use client"
import { useState } from "react"
import { signIn, signUp, AuthUser } from "@/lib/datasense-api"

interface Props {
  onClose: () => void
  onSuccess: (user: AuthUser) => void
  initialMode?: "login" | "register"
}

export default function AuthModal({ onClose, onSuccess, initialMode = "login" }: Props) {
  const [mode, setMode]       = useState<"login" | "register">(initialMode)
  const [email, setEmail]     = useState("")
  const [password, setPass]   = useState("")
  const [name, setName]       = useState("")
  const [error, setError]     = useState("")
  const [busy, setBusy]       = useState(false)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(""); setBusy(true)
    try {
      const res = mode === "login"
        ? await signIn(email, password)
        : await signUp(email, password, name || undefined)
      onSuccess(res.user)
    } catch (err: any) { setError(err.message) }
    finally { setBusy(false) }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(4,7,15,0.85)", backdropFilter: "blur(10px)" }}
      onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="w-full max-w-sm rounded-2xl p-8 fade-up"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>

        <h2 className="text-xl font-bold mb-1" style={{ fontFamily: "Syne, sans-serif" }}>
          {mode === "login" ? "Welcome back" : "Create account"}
        </h2>
        <p className="text-sm mb-6" style={{ color: "var(--muted)" }}>
          {mode === "login" ? "Sign in to access your history across devices." : "Save your analyses anywhere."}
        </p>

        <form onSubmit={submit} className="space-y-4">
          {mode === "register" && (
            <Field label="Name (optional)" type="text" value={name} onChange={setName} placeholder="Your name" />
          )}
          <Field label="Email" type="email" value={email} onChange={setEmail} placeholder="you@example.com" required />
          <Field label="Password" type="password" value={password} onChange={setPass} placeholder="Min 8 characters" required />

          {error && (
            <p className="text-sm px-3 py-2 rounded-lg" style={{ color: "var(--critical)", background: "rgba(239,68,68,0.08)" }}>
              {error}
            </p>
          )}

          <button type="submit" disabled={busy}
            className="w-full py-2.5 rounded-xl font-semibold text-sm transition-opacity disabled:opacity-50"
            style={{ background: "var(--teal)", color: "#070D1A", fontFamily: "Syne, sans-serif" }}>
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>

        <p className="mt-4 text-sm text-center" style={{ color: "var(--muted)" }}>
          {mode === "login" ? "No account? " : "Already have one? "}
          <button onClick={() => { setMode(mode === "login" ? "register" : "login"); setError("") }}
            className="underline" style={{ color: "var(--teal)" }}>
            {mode === "login" ? "Sign up" : "Sign in"}
          </button>
        </p>
      </div>
    </div>
  )
}

function Field({ label, type, value, onChange, placeholder, required }: {
  label: string; type: string; value: string; onChange: (v: string) => void; placeholder: string; required?: boolean
}) {
  return (
    <div>
      <label className="block text-xs font-medium mb-1.5" style={{ color: "var(--dim)" }}>{label}</label>
      <input type={type} value={value} onChange={e => onChange(e.target.value)}
        placeholder={placeholder} required={required}
        className="w-full px-3 py-2.5 rounded-lg border text-sm outline-none"
        style={{ background: "var(--surface-2)", borderColor: "var(--border)", color: "var(--text)" }} />
    </div>
  )
}

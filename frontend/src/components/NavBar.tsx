"use client"
import { useState, useEffect } from "react"
import Link from "next/link"
import { fetchCurrentUser, signOut, AuthUser } from "@/lib/datasense-api"
import AuthModal from "./AuthModal"

export default function NavBar() {
  const [user, setUser]         = useState<AuthUser | null>(null)
  const [showAuth, setShowAuth] = useState<false | "login" | "register">(false)
  const [showMenu, setShowMenu] = useState(false)

  useEffect(() => { fetchCurrentUser().then(r => setUser(r.user)) }, [])

  const handleSignOut = async () => { await signOut(); setUser(null); setShowMenu(false) }

  return (
    <>
      <header className="fixed top-0 inset-x-0 z-40 h-14 flex items-center px-6 border-b"
        style={{ background: "rgba(7,13,26,0.88)", backdropFilter: "blur(14px)", borderColor: "var(--border)" }}>

        <Link href="/" className="flex items-center gap-2 mr-8">
          <span className="w-7 h-7 rounded-md flex items-center justify-center text-xs font-bold"
            style={{ background: "var(--teal)", color: "#070D1A", fontFamily: "Syne, sans-serif" }}>DS</span>
          <span className="font-semibold" style={{ fontFamily: "Syne, sans-serif" }}>DataSense</span>
        </Link>

        <nav className="flex items-center gap-5 flex-1">
          <Link href="/" className="text-sm transition-colors hover:text-teal-400" style={{ color: "var(--muted)" }}>Analyze</Link>
          <Link href="/history" className="text-sm transition-colors hover:text-teal-400" style={{ color: "var(--muted)" }}>History</Link>
        </nav>

        {user ? (
          <div className="relative">
            <button onClick={() => setShowMenu(!showMenu)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm"
              style={{ background: "var(--surface-2)", color: "var(--dim)" }}>
              <span className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                style={{ background: "var(--teal)", color: "#070D1A" }}>
                {(user.name || user.email)[0].toUpperCase()}
              </span>
              {user.name || user.email.split("@")[0]}
            </button>
            {showMenu && (
              <>
                <div className="fixed inset-0 z-30" onClick={() => setShowMenu(false)} />
                <div className="absolute right-0 top-full mt-2 w-44 rounded-xl border py-1 z-40 shadow-xl text-sm"
                  style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
                  <div className="px-3 py-2 border-b text-xs truncate" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>{user.email}</div>
                  <button onClick={handleSignOut} className="w-full text-left px-3 py-2 transition-colors hover:text-red-400" style={{ color: "var(--dim)" }}>Sign out</button>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <button onClick={() => setShowAuth("login")} className="text-sm" style={{ color: "var(--muted)" }}>Sign in</button>
            <button onClick={() => setShowAuth("register")}
              className="px-3 py-1.5 rounded-lg text-sm font-semibold"
              style={{ background: "var(--teal)", color: "#070D1A", fontFamily: "Syne, sans-serif" }}>
              Sign up
            </button>
          </div>
        )}
      </header>

      {showAuth && <AuthModal initialMode={showAuth} onClose={() => setShowAuth(false)} onSuccess={u => { setUser(u); setShowAuth(false) }} />}
    </>
  )
}

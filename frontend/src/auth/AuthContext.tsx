import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api/client'

type Branding = { display_name: string; logo_url: string; accent_color: string }

type AuthState = {
  token: string | null
  me: any | null
  branding: Branding | null
  refresh: () => Promise<void>
  logout: () => void
  setToken: (t: string) => void
}

const AuthCtx = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState<string | null>(() => localStorage.getItem('token'))
  const [me, setMe] = useState<any | null>(null)
  const [branding, setBranding] = useState<Branding | null>(null)

  const setToken = (t: string) => {
    localStorage.setItem('token', t)
    setTokenState(t)
  }

  const logout = () => {
    localStorage.removeItem('token')
    setTokenState(null)
    setMe(null)
  }

  const refresh = async () => {
    if (!localStorage.getItem('token')) return
    const [meRes, brandRes] = await Promise.all([api.me(), api.branding().catch(() => null)])
    const normalized = {
      email: meRes.user?.email || (meRes as any).email,
      name: meRes.user?.name || (meRes as any).name,
      organization_id:
        meRes.organization?.organization_id ||
        (meRes as any).organization_id ||
        meRes.organizations?.[0]?.organization_id,
      organizations: meRes.organizations || [],
    }
    setMe(normalized)
    if (brandRes) {
      setBranding(brandRes)
      document.documentElement.style.setProperty('--accent', brandRes.accent_color || '#0F766E')
      document.documentElement.style.setProperty(
        '--accent-soft',
        `${brandRes.accent_color || '#0F766E'}33`,
      )
    }
  }

  useEffect(() => {
    if (token) refresh().catch(() => logout())
  }, [token])

  const value = useMemo(
    () => ({ token, me, branding, refresh, logout, setToken }),
    [token, me, branding],
  )

  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthCtx)
  if (!ctx) throw new Error('useAuth outside provider')
  return ctx
}

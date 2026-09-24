import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AuthProvider, useAuth } from './auth/AuthContext'
import { CampaignProvider } from './campaign/CampaignContext'
import { AppShell } from './components/AppShell'
import { LoginPage, SignupPage } from './pages/AuthPages'
import { PrivacyPolicyPage, TermsOfServicePage } from './pages/LegalPages'
import { HomePage } from './pages/HomePage'
import { InboxPage } from './pages/InboxPage'
import { ThreadPage } from './pages/ThreadPage'
import { CampaignSetupPage } from './pages/campaigns/CampaignSetupPage'
import { CampaignLivePage } from './pages/campaigns/CampaignLivePage'
import { CataloguesPage } from './pages/CataloguesPage'
import { PublicCataloguePage } from './pages/PublicCataloguePage'
import { PromptsPage } from './pages/PromptsPage'
import { TemplatesPage } from './pages/TemplatesPage'
import { AiCreateTemplatePage } from './pages/AiCreateTemplatePage'
import { RagLabPage } from './pages/RagLabPage'
import { SettingsPage } from './pages/SettingsPage'
import { UsersPage } from './pages/UsersPage'
import type { ReactNode } from 'react'

const qc = new QueryClient()

function Protected({ children }: { children: ReactNode }) {
  const { token } = useAuth()
  if (!token) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/privacy" element={<PrivacyPolicyPage />} />
            <Route path="/terms" element={<TermsOfServicePage />} />
            <Route path="/c/:orgSlug/:catalogueSlug" element={<PublicCataloguePage />} />
            <Route
              path="/"
              element={
                <Protected>
                  <CampaignProvider>
                    <AppShell />
                  </CampaignProvider>
                </Protected>
              }
            >
              <Route index element={<HomePage />} />
              <Route path="inbox" element={<InboxPage />} />
              <Route path="inbox/:id" element={<ThreadPage />} />
              <Route path="campaigns" element={<CampaignSetupPage />} />
              <Route path="campaigns/review" element={<Navigate to="/campaigns/live" replace />} />
              <Route path="campaigns/live" element={<CampaignLivePage />} />
              <Route path="campaigns/new/*" element={<Navigate to="/campaigns" replace />} />
              <Route path="catalogues" element={<CataloguesPage />} />
              <Route path="knowledge" element={<Navigate to="/catalogues" replace />} />
              <Route path="settings" element={<SettingsPage />} />
              <Route path="admin/users" element={<UsersPage />} />
              <Route path="admin/prompts" element={<PromptsPage />} />
              <Route path="admin/templates" element={<TemplatesPage />} />
              <Route path="admin/templates/create" element={<AiCreateTemplatePage />} />
              <Route path="admin/rag" element={<RagLabPage />} />
              <Route path="prompts" element={<Navigate to="/admin/prompts" replace />} />
              <Route path="templates" element={<Navigate to="/admin/templates" replace />} />
              <Route path="templates/create" element={<Navigate to="/admin/templates/create" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  )
}

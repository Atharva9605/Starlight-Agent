import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AuthProvider, useAuth } from './auth/AuthContext'
import { CampaignProvider } from './campaign/CampaignContext'
import { AppShell } from './components/AppShell'
import { LoginPage, SignupPage } from './pages/AuthPages'
import { HomePage } from './pages/HomePage'
import { InboxPage } from './pages/InboxPage'
import { ThreadPage } from './pages/ThreadPage'
import { CampaignsOverviewPage } from './pages/campaigns/CampaignsOverviewPage'
import { NewCampaignLayout } from './pages/campaigns/NewCampaignLayout'
import { StepLeads } from './pages/campaigns/StepLeads'
import { StepDesign } from './pages/campaigns/StepDesign'
import { StepReview } from './pages/campaigns/StepReview'
import { LiveRunPage } from './pages/campaigns/LiveRunPage'
import { CataloguesPage } from './pages/CataloguesPage'
import { PromptsPage } from './pages/PromptsPage'
import { TemplatesPage } from './pages/TemplatesPage'
import { RagLabPage } from './pages/RagLabPage'
import { SettingsPage } from './pages/SettingsPage'
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
              <Route path="campaigns" element={<CampaignsOverviewPage />} />
              <Route path="campaigns/new" element={<NewCampaignLayout />}>
                <Route index element={<Navigate to="/campaigns/new/leads" replace />} />
                <Route path="leads" element={<StepLeads />} />
                <Route path="design" element={<StepDesign />} />
                <Route path="review" element={<StepReview />} />
              </Route>
              <Route path="campaigns/live" element={<LiveRunPage />} />
              <Route path="catalogues" element={<CataloguesPage />} />
              <Route path="knowledge" element={<Navigate to="/catalogues" replace />} />
              <Route path="settings" element={<SettingsPage />} />
              <Route path="admin/prompts" element={<PromptsPage />} />
              <Route path="admin/templates" element={<TemplatesPage />} />
              <Route path="admin/rag" element={<RagLabPage />} />
              <Route path="prompts" element={<Navigate to="/admin/prompts" replace />} />
              <Route path="templates" element={<Navigate to="/admin/templates" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  )
}

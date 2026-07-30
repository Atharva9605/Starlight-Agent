import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useCampaign } from '../../campaign/CampaignContext'

const STEPS = [
  { to: '/campaigns/new/leads', label: 'Leads', n: 1 },
  { to: '/campaigns/new/design', label: 'Design', n: 2 },
  { to: '/campaigns/new/review', label: 'Review', n: 3 },
]

export function NewCampaignLayout() {
  const { leads } = useCampaign()
  const { pathname } = useLocation()
  const activeIdx = STEPS.findIndex((s) => pathname.startsWith(s.to))

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>New campaign</h1>
          <p>Three steps: add your leads, choose the look, then launch.</p>
        </div>
      </div>

      <ol className="stepper">
        {STEPS.map((s, i) => {
          const locked = i > 0 && leads.length === 0
          const complete = i < activeIdx
          return (
            <li key={s.to} className={`stepper-item${complete ? ' complete' : ''}`}>
              <NavLink
                to={locked ? '/campaigns/new/leads' : s.to}
                className={({ isActive }) =>
                  `stepper-link${isActive ? ' active' : ''}${locked ? ' locked' : ''}`
                }
              >
                <span className="stepper-dot">{complete ? '✓' : s.n}</span>
                <span>{s.label}</span>
              </NavLink>
              {i < STEPS.length - 1 ? <span className="stepper-bar" /> : null}
            </li>
          )
        })}
      </ol>

      <Outlet />
    </div>
  )
}

import type { ReactNode } from 'react'
import { BeakerRegular, CalendarClockRegular, CheckmarkCircleFilled, EyeRegular, SendRegular } from '@fluentui/react-icons'
import { useCampaign } from '../../../campaign/CampaignContext'
import { CardHeader, MessageBar, Switch } from '../../../components/ui'
import { WizardNav, formatScheduled, toLocalInput, useWizard, type SendMode } from './CampaignWizard'

const MODES: { value: SendMode; label: string; blurb: string; icon: ReactNode }[] = [
  {
    value: 'review',
    label: 'Review each email first',
    blurb: 'Every email is written, then you read, edit, send or discard each one. Best for a new list.',
    icon: <EyeRegular />,
  },
  {
    value: 'auto',
    label: 'Send automatically now',
    blurb: 'Each email sends as soon as it is written. Watch progress live, pause any time.',
    icon: <SendRegular />,
  },
  {
    value: 'schedule',
    label: 'Schedule for later',
    blurb: 'The server starts at the time you pick and sends automatically. You can close the browser.',
    icon: <CalendarClockRegular />,
  },
]

/** Step 3 — how and when emails go out, plus test mode. */
export function SendingStep() {
  const { recipientOverride, setRecipientOverride, delay, setDelay, attachProductSheet, setAttachProductSheet } = useCampaign()
  const { mode, setMode, scheduleAt, setScheduleAt, scheduleValid } = useWizard()

  return (
    <>
      <div className="wizard-body">
        <section className="card">
          <CardHeader title="How should emails go out?" />
          <div className="stack tight" role="radiogroup" aria-label="How emails go out" style={{ marginTop: 16 }}>
            {MODES.map((m) => (
              <button
                key={m.value}
                type="button"
                role="radio"
                aria-checked={mode === m.value}
                className={`choice-card${mode === m.value ? ' selected' : ''}`}
                onClick={() => setMode(m.value)}
              >
                <span className="choice-card-icon">{m.icon}</span>
                <span className="choice-card-copy">
                  <strong>{m.label}</strong>
                  <span>{m.blurb}</span>
                </span>
                {mode === m.value ? <CheckmarkCircleFilled className="choice-card-check" /> : <span className="choice-card-ring" />}
              </button>
            ))}
          </div>
          {mode === 'schedule' ? (
            <label className="field" style={{ marginTop: 16 }}>
              <span>Start at (your local time)</span>
              <input
                className="input"
                type="datetime-local"
                min={toLocalInput(new Date())}
                value={scheduleAt}
                onChange={(e) => setScheduleAt(e.target.value)}
              />
              {scheduleValid ? (
                <span className="field-hint">Starts {formatScheduled(scheduleAt)}</span>
              ) : (
                <span className="field-hint text-danger">Pick a time in the future.</span>
              )}
            </label>
          ) : null}
        </section>

        <aside className="stack loose">
          <section className="card">
            <CardHeader icon={<BeakerRegular />} title="Test first" subtitle="Send everything to your own inbox instead of the leads." />
            <label className="field" style={{ marginTop: 12 }}>
              <span>Test inbox</span>
              <input
                className="input"
                type="email"
                placeholder="you@starlightlinearled.com"
                value={recipientOverride}
                onChange={(e) => setRecipientOverride(e.target.value)}
              />
              <span className="field-hint">Leave empty to email the real leads.</span>
            </label>
            {recipientOverride.trim() ? (
              <MessageBar intent="warning" title="Test mode is on" className="mt-12">
                No lead will be contacted. Every email goes to {recipientOverride.trim()}.
              </MessageBar>
            ) : null}
          </section>

          <section className="card">
            <CardHeader title="Options" />
            <div style={{ marginTop: 4 }}>
              <Switch
                checked={attachProductSheet}
                onChange={setAttachProductSheet}
                label="Attach product sheet PDF"
                description="A branded PDF of the catalogue products suggested in each email."
              />
            </div>
            <div className="field" style={{ marginTop: 8 }}>
              <span className="field-label">Pause between sends</span>
              <div className="slider-row">
                <input type="range" min={1} max={15} value={delay} aria-label="Seconds between sends" onChange={(e) => setDelay(Number(e.target.value))} />
                <span className="slider-value">{delay}s</span>
              </div>
              <span className="field-hint">A short gap keeps sending gentle on your mailbox.</span>
            </div>
          </section>
        </aside>
      </div>
      <WizardNav step={2} canNext={mode !== 'schedule' || scheduleValid} />
    </>
  )
}

import { useState, type ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  BeakerRegular,
  CalendarClockRegular,
  DocumentTableRegular,
  EditRegular,
  EyeRegular,
  PaintBrushRegular,
  PlayRegular,
  SendRegular,
} from '@fluentui/react-icons'
import { useCampaign } from '../../../campaign/CampaignContext'
import { MessageBar, Spinner, useConfirm } from '../../../components/ui'
import { WizardNav, formatScheduled, stepUrl, useWizard } from './CampaignWizard'

function SummaryRow({ icon, label, value, detail, edit }: { icon: ReactNode; label: string; value: ReactNode; detail?: ReactNode; edit: number }) {
  return (
    <div className="summary-row">
      <span className="summary-icon">{icon}</span>
      <div className="grow" style={{ minWidth: 0 }}>
        <span className="summary-label">{label}</span>
        <strong className="summary-value">{value}</strong>
        {detail ? <span className="summary-detail">{detail}</span> : null}
      </div>
      <Link to={stepUrl(edit)} className="btn subtle sm">
        <EditRegular /> Change
      </Link>
    </div>
  )
}

/** Step 4 — one last look, then start. */
export function LaunchStep() {
  const nav = useNavigate()
  const confirm = useConfirm()
  const { leads, fileName, template, recipientOverride, attachProductSheet, delay, start, schedule, status } = useCampaign()
  const { templateOptions, mode, scheduleAt, scheduleValid, noMailbox } = useWizard()
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState('')

  const inFlight = status === 'running' || status === 'reviewing' || status === 'paused'
  const lookups = leads.filter((l) => !l.website && (l.company || l.name)).length
  const testInbox = recipientOverride.trim()
  const n = leads.length
  const emails = `${n} email${n === 1 ? '' : 's'}`

  const launch = async () => {
    setError('')
    if (mode !== 'review' && !testInbox) {
      const ok = await confirm({
        title: mode === 'schedule' ? `Schedule ${emails}?` : `Send ${emails} now?`,
        body: `These go straight to real leads without a review step${mode === 'schedule' ? `, starting ${formatScheduled(scheduleAt)}` : ''}.`,
        confirmLabel: mode === 'schedule' ? 'Schedule' : 'Start sending',
      })
      if (!ok) return
    }
    setLaunching(true)
    try {
      if (mode === 'schedule') {
        if (!scheduleValid) throw new Error('Pick a start time in the future.')
        await schedule(new Date(scheduleAt))
        nav('/campaigns/runs')
        return
      }
      const dest = await start()
      nav(dest === 'live' ? '/campaigns/live' : '/campaigns/review')
    } catch (e: any) {
      setError(e?.message || 'Could not start the campaign')
    } finally {
      setLaunching(false)
    }
  }

  const modeText =
    mode === 'review'
      ? 'Review each email first'
      : mode === 'auto'
        ? 'Send automatically now'
        : `Scheduled for ${formatScheduled(scheduleAt) || '—'}`
  const modeDetail =
    mode === 'review'
      ? 'Nothing is sent until you approve it on the review board.'
      : mode === 'auto'
        ? `Each email sends as soon as it is written, ${delay}s apart.`
        : 'The server starts on its own. You can close the browser.'

  const label = launching
    ? 'Starting…'
    : mode === 'schedule'
      ? `Schedule ${emails}`
      : mode === 'auto'
        ? `Send ${emails}`
        : `Write ${emails} for review`

  return (
    <>
      <section className="card summary">
        <SummaryRow
          icon={<DocumentTableRegular />}
          label="Leads"
          value={`${n} lead${n === 1 ? '' : 's'}`}
          detail={`${fileName}${lookups ? ` · ${lookups} need a website lookup` : ''}`}
          edit={0}
        />
        <SummaryRow
          icon={<PaintBrushRegular />}
          label="Design"
          value={templateOptions.find((t) => t.value === template)?.label || template}
          detail={attachProductSheet ? 'With a product sheet PDF attached' : 'No attachment'}
          edit={1}
        />
        <SummaryRow
          icon={mode === 'schedule' ? <CalendarClockRegular /> : mode === 'auto' ? <SendRegular /> : <EyeRegular />}
          label="Sending"
          value={modeText}
          detail={modeDetail}
          edit={2}
        />
        <SummaryRow
          icon={<BeakerRegular />}
          label="Recipients"
          value={testInbox ? `Test mode: everything goes to ${testInbox}` : 'The real leads'}
          detail={testInbox ? 'No lead will be contacted.' : 'Each lead receives their own email.'}
          edit={2}
        />
      </section>

      <div className="page-alerts" style={{ marginTop: 16 }}>
        {noMailbox && mode !== 'review' ? (
          <MessageBar intent="error" title="Connect Gmail before sending">
            Automatic and scheduled campaigns need a connected mailbox. Choose “Review each email first” or connect Gmail in Settings.
          </MessageBar>
        ) : null}
        {error ? (
          <MessageBar intent="error" title="Couldn't start" onDismiss={() => setError('')}>
            {error}
          </MessageBar>
        ) : null}
      </div>

      <WizardNav
        step={3}
        nextHint={testInbox ? 'Test mode' : mode === 'review' ? 'You approve every email' : 'Sends to real leads'}
        primary={
          <button
            type="button"
            className="btn lg"
            disabled={launching || inFlight || !n || (mode === 'schedule' && !scheduleValid) || (noMailbox && mode !== 'review')}
            onClick={() => void launch()}
          >
            {launching ? <Spinner size="sm" /> : mode === 'schedule' ? <CalendarClockRegular /> : mode === 'auto' ? <SendRegular /> : <PlayRegular />}
            {label}
          </button>
        }
      />
    </>
  )
}

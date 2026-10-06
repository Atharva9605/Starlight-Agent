/**
 * The conversation list API returns flat client fields (client_company, client_email)
 * plus draft_count / needs_action. Read them here so every page agrees.
 */

export function convCompany(c: any): string {
  return c.client_company || c.client?.company || c.client_email || c.client?.email || 'Client'
}

export function convEmail(c: any): string {
  return c.client_email || c.client?.email || ''
}

/** An AI draft is waiting for a person to approve it. */
export function convNeedsReview(c: any): boolean {
  return Number(c.draft_count || 0) > 0 || Boolean(c.has_draft || c.pending_draft)
}

/** The client wrote last and nobody has drafted a reply yet. */
export function convClientWaiting(c: any): boolean {
  if (convNeedsReview(c)) return false
  return c.last_direction === 'inbound' || (c.needs_action === true && !convNeedsReview(c))
}

export function convSnippet(c: any): string {
  return String(c.last_message_preview || c.snippet || '').replace(/\s+/g, ' ').trim()
}

export function convBadge(c: any): { label: string; cls: string } {
  if (convNeedsReview(c)) return { label: 'Draft to review', cls: 'warning' }
  if (convClientWaiting(c)) return { label: 'Client waiting', cls: 'danger' }
  if (c.last_direction === 'outbound') return { label: 'Replied', cls: 'success' }
  return { label: 'Open', cls: '' }
}

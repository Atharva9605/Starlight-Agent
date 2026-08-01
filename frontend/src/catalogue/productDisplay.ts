/**
 * Normalize messy RAG / text-extract catalogue product payloads into
 * clean fields suitable for a digital catalogue UI.
 */

const EMPTY = /^(["']?(not stated|n\/a|na|none|unknown|-|—|–|\.?)["']?)$/i

const SPEC_LABELS: Record<string, string> = {
  code: 'Code',
  mounting: 'Mounting',
  ip: 'IP rating',
  ip_rating: 'IP rating',
  cct: 'CCT',
  wattage: 'Wattage',
  beam: 'Beam',
  beam_angle: 'Beam',
  diffuser: 'Diffuser',
  cri: 'CRI',
  input: 'Input',
  applications: 'Applications',
  notes: 'Notes',
  voltage: 'Voltage',
  lumen: 'Lumen',
  dimensions: 'Dimensions',
  finish: 'Finish',
  material: 'Material',
}

function isEmpty(v: unknown): boolean {
  if (v == null) return true
  const s = String(v).trim()
  return !s || EMPTY.test(s)
}

function cleanValue(v: unknown): string {
  return String(v ?? '').trim()
}

/** Unwrap accidental JSON-array / quoted wrappers from text extract. */
export function unwrapDocument(raw: string): string {
  let text = (raw || '').trim()
  if (!text) return ''

  // ["foo | bar", "baz"]  or  "foo | bar"
  if (text.startsWith('[') || text.startsWith('"')) {
    try {
      const parsed = JSON.parse(text.startsWith('[') || text.startsWith('"') ? text : text)
      if (Array.isArray(parsed)) {
        text = parsed.map((x) => String(x)).join('\n')
      } else if (typeof parsed === 'string') {
        text = parsed
      }
    } catch {
      text = text.replace(/^\[+|\]+$/g, '').replace(/^"+|"+$/g, '')
    }
  }
  return text.trim()
}

function labelKey(label: string): string {
  return label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '')
}

/**
 * Parse pipe-delimited extract lines into name + specs.
 * Example: `HyperSlot | Code: SLL | 57.5 x 24 mm | Mounting: surface | IP: IP40`
 */
export function parseProductText(raw: string): {
  name: string
  specs: Record<string, string>
  extras: string[]
} {
  const text = unwrapDocument(raw)
  if (!text) return { name: '', specs: {}, extras: [] }

  // Prefer first line / first product if multiple were jammed together
  const primary = text.split(/\n+/).map((l) => l.trim()).filter(Boolean)[0] || text
  const parts = primary.split('|').map((p) => p.trim()).filter(Boolean)

  let name = ''
  const specs: Record<string, string> = {}
  const extras: string[] = []

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]
    const m = part.match(/^([^:]+):\s*(.+)$/)
    if (m) {
      const key = labelKey(m[1])
      const val = cleanValue(m[2])
      if (!isEmpty(val)) specs[key] = val
      continue
    }
    if (i === 0 && !name) {
      name = part
      continue
    }
    // Bare dimension-like token → Dimensions
    if (/^\d+(\.\d+)?\s*x\s*\d+/i.test(part) || /\d+\s*mm/i.test(part)) {
      if (!specs.dimensions) specs.dimensions = part
      else extras.push(part)
      continue
    }
    if (!isEmpty(part)) extras.push(part)
  }

  return { name: name.trim(), specs, extras }
}

export type DisplaySpec = { key: string; label: string; value: string }

export type DisplayProduct = {
  id: string
  name: string
  category: string
  imageUrl: string
  pageNumber: number
  description: string
  features: string[]
  specs: DisplaySpec[]
  specsPreview: string
  code: string
}

export function toDisplayProduct(p: {
  id: string
  product_name: string
  category?: string
  description?: string
  features?: string[]
  specs?: Record<string, string>
  variants?: string[]
  page_number?: number
  image_url?: string
  specs_preview?: string
}): DisplayProduct {
  const fromDb = Object.fromEntries(
    Object.entries(p.specs || {})
      .map(([k, v]) => [labelKey(k), cleanValue(v)])
      .filter(([, v]) => !isEmpty(v)),
  )

  const parsed = parseProductText(p.description || '')
  const merged: Record<string, string> = { ...parsed.specs, ...fromDb }

  // Prefer clean name: DB name if not junk, else parsed
  let name = cleanValue(p.product_name)
  if (!name || name.length > 80 || name.includes('|') || name.startsWith('[')) {
    name = parsed.name || name || 'Product'
  }
  // If DB name is first token of pipe string, parsed name is better when longer context missing
  if (name.includes('|')) name = parseProductText(name).name || name.split('|')[0].trim()

  const features = (p.features || [])
    .map((f) => cleanValue(f))
    .filter((f) => !isEmpty(f))

  // Prefer a short human description without the pipe dump
  let description = ''
  const rawDesc = unwrapDocument(p.description || '')
  if (rawDesc && !rawDesc.includes('|') && rawDesc.length < 280) {
    description = rawDesc
  } else if (merged.applications && !isEmpty(merged.applications)) {
    description = merged.applications
  } else if (merged.notes && !isEmpty(merged.notes)) {
    description = merged.notes
  }

  const order = [
    'code',
    'dimensions',
    'wattage',
    'cct',
    'cri',
    'ip',
    'ip_rating',
    'beam',
    'beam_angle',
    'voltage',
    'input',
    'mounting',
    'diffuser',
    'lumen',
    'finish',
    'material',
    'applications',
    'notes',
  ]

  const specs: DisplaySpec[] = []
  const seen = new Set<string>()
  for (const key of order) {
    if (merged[key] && !seen.has(key)) {
      seen.add(key)
      specs.push({
        key,
        label: SPEC_LABELS[key] || key.replace(/_/g, ' '),
        value: merged[key],
      })
    }
  }
  for (const [key, value] of Object.entries(merged)) {
    if (seen.has(key) || key === 'applications' || key === 'notes' || key === 'anomaly') continue
    specs.push({
      key,
      label: SPEC_LABELS[key] || key.replace(/_/g, ' '),
      value,
    })
  }

  // Drop applications/notes from table if already used as description
  const tableSpecs = specs.filter(
    (s) => !(description && (s.key === 'applications' || s.key === 'notes')),
  )

  const previewParts = tableSpecs
    .filter((s) => ['wattage', 'cct', 'cri', 'ip', 'ip_rating', 'dimensions', 'beam'].includes(s.key))
    .slice(0, 4)
    .map((s) => s.value)

  const category = cleanValue(p.category || '')
  const showCategory = category && category.toLowerCase() !== 'other'

  return {
    id: p.id,
    name,
    category: showCategory ? category : '',
    imageUrl: cleanValue(p.image_url || ''),
    pageNumber: p.page_number || 0,
    description,
    features,
    specs: tableSpecs.slice(0, 8),
    specsPreview: cleanValue(p.specs_preview) || previewParts.join(' · '),
    code: merged.code || '',
  }
}

export function cleanCatalogueTitle(name: string): string {
  return cleanValue(name)
    .replace(/\.pdf$/i, '')
    .replace(/[_-]+/g, ' ')
    .trim()
}

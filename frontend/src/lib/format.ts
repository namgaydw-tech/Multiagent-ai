/** Formatting helpers: metric records → display strings (never invent values). */

import type { MetricRecord } from './types'

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a'
  return `${(value * 100).toFixed(digits)}%`
}

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a'
  return value.toFixed(digits)
}

/** "0.051 [0.029–0.077]" — undefined metrics render as "n/a", never as 0. */
export function metricWithCI(rec: MetricRecord | undefined | null): string {
  if (!rec || rec.value === null || rec.value === undefined) return 'n/a'
  const ci = rec.ci95
  const value = typeof rec.value === 'number' ? rec.value : Number(rec.value)
  if (!ci || ci[0] === null || ci[1] === null) return num(value)
  return `${num(value)} [${num(ci[0])}, ${num(ci[1])}]`
}

export function metricN(rec: MetricRecord | undefined | null): string {
  if (!rec || !rec.n_den) return ''
  return `n=${rec.n_den}`
}

export function pValue(p: number | null | undefined): string {
  if (p === null || p === undefined) return 'n/a'
  if (p === 0) return '<1e-16'
  if (p < 0.001) return p.toExponential(2)
  return p.toFixed(4)
}

export function shortHash(hash: string | null | undefined): string {
  if (!hash) return 'n/a'
  return hash.length > 12 ? `${hash.slice(0, 12)}…` : hash
}

export function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

/** Pick the display value of a headline metric record. */
export function headlineValue(headline: Record<string, unknown> | undefined, key: string): number | null {
  const rec = headline?.[key] as MetricRecord | undefined
  if (!rec || rec.value === null || rec.value === undefined) return null
  return typeof rec.value === 'number' ? rec.value : Number(rec.value)
}

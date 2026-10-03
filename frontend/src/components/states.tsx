import { AlertTriangle, Inbox, Loader2 } from 'lucide-react'
import type { ReactNode } from 'react'
import { ApiError } from '../lib/api'

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400"
         role="status" aria-live="polite">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      {label}
    </div>
  )
}

/** Real error message + actionable hint from the backend (never a blank page). */
export function ErrorBox({ error, onRetry }: { error: ApiError | Error; onRetry?: () => void }) {
  const apiErr = error instanceof ApiError ? error : null
  return (
    <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
         role="alert">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        <div className="min-w-0">
          <p className="font-semibold">Something went wrong</p>
          <p className="mt-1 break-words">{error.message}</p>
          {apiErr?.hint && (
            <p className="mt-2 rounded bg-red-100/70 px-2 py-1 font-mono text-xs dark:bg-red-900/50">
              {apiErr.hint}
            </p>
          )}
          {onRetry && (
            <button className="btn-secondary mt-3" onClick={onRetry}>
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/** Explicit empty state for missing artifacts (available: false + hint). */
export function EmptyState({
  title,
  reason,
  hint,
}: {
  title: string
  reason?: string
  hint?: string
}) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-slate-300 p-8 text-center dark:border-slate-700">
      <Inbox className="h-8 w-8 text-slate-400" aria-hidden />
      <p className="font-medium text-slate-600 dark:text-slate-300">{title}</p>
      {reason && <p className="text-sm text-slate-500 dark:text-slate-400">{reason}</p>}
      {hint && (
        <code className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          {hint}
        </code>
      )}
    </div>
  )
}

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            {title && <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100">{title}</h2>}
            {subtitle && <p className="text-xs text-slate-500 dark:text-slate-400">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

export function MetricTile({
  label,
  value,
  sub,
  tone = 'default',
}: {
  label: string
  value: string
  sub?: string
  tone?: 'default' | 'good' | 'warn' | 'bad'
}) {
  const tones = {
    default: 'text-slate-800 dark:text-slate-100',
    good: 'text-emerald-600 dark:text-emerald-400',
    warn: 'text-amber-600 dark:text-amber-400',
    bad: 'text-red-600 dark:text-red-400',
  }
  return (
    <div className="card flex flex-col gap-1">
      <span className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </span>
      <span className={`text-lg font-bold tabular-nums break-all sm:text-2xl ${tones[tone]}`}>{value}</span>
      {sub && <span className="text-xs text-slate-500 dark:text-slate-400">{sub}</span>}
    </div>
  )
}

export function Badge({
  children,
  tone = 'slate',
}: {
  children: ReactNode
  tone?: 'slate' | 'green' | 'amber' | 'red' | 'blue'
}) {
  const tones = {
    slate: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
    green: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-300',
    amber: 'bg-amber-100 text-amber-800 dark:bg-amber-900/60 dark:text-amber-300',
    red: 'bg-red-100 text-red-800 dark:bg-red-900/60 dark:text-red-300',
    blue: 'bg-blue-100 text-blue-800 dark:bg-blue-900/60 dark:text-blue-300',
  }
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  )
}

export function PageHeader({ title, description }: { title: string; description?: string }) {
  return (
    <header className="mb-4">
      <h1 className="text-xl font-bold text-slate-900 dark:text-white sm:text-2xl">{title}</h1>
      {description && <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{description}</p>}
    </header>
  )
}

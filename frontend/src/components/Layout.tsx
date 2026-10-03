import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import {
  Activity,
  BarChart3,
  Beaker,
  BrainCircuit,
  FlaskConical,
  ListChecks,
  Menu,
  Moon,
  Search,
  ShieldAlert,
  Sun,
  TerminalSquare,
  X,
} from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useAsync } from '../hooks/useAsync'
import { api } from '../lib/api'

const NAV = [
  { to: '/', label: 'Overview', icon: Activity, end: true },
  { to: '/model', label: 'Model Performance', icon: BarChart3 },
  { to: '/cases', label: 'Case Explorer', icon: Search },
  { to: '/agents', label: 'Agent Timeline', icon: BrainCircuit },
  { to: '/bias', label: 'Bias Experiments', icon: FlaskConical },
  { to: '/ablations', label: 'Ablations', icon: Beaker },
  { to: '/errors', label: 'Error Analysis', icon: ShieldAlert },
  { to: '/audits', label: 'Audit Viewer', icon: ListChecks },
  { to: '/repro', label: 'Reproduction', icon: TerminalSquare },
]

/** App shell: responsive sidebar (drawer < lg), header with theme toggle + backend dot. */
export function Layout() {
  const { theme, toggle } = useTheme()
  const [open, setOpen] = useState(false)
  const { data: health } = useAsync(() => api.health(), [])
  const backendOk = health?.status === 'ok'

  const nav = (
    <nav className="flex flex-col gap-1 p-3" aria-label="Main navigation">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${
              isActive
                ? 'bg-brand-600 text-white shadow-sm'
                : 'text-slate-600 hover:bg-slate-200 dark:text-slate-300 dark:hover:bg-slate-800'
            }`
          }
        >
          <Icon className="h-4 w-4 shrink-0" aria-hidden />
          {label}
        </NavLink>
      ))}
    </nav>
  )

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-900/90">
        <div className="flex h-14 items-center gap-3 px-3 sm:px-4">
          <button
            className="btn-secondary !px-2 lg:hidden"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? 'Close menu' : 'Open menu'}
          >
            {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-bold text-slate-900 dark:text-white sm:text-base">
              Pediatric DX · Confirmation-Bias Research
            </p>
            <p className="hidden truncate text-xs text-slate-500 dark:text-slate-400 sm:block">
              Adversarial multi-agent AI — research prototype, not a medical device
            </p>
          </div>
          <span
            className={`hidden items-center gap-1.5 rounded-full px-2 py-1 text-xs sm:inline-flex ${
              backendOk
                ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-300'
                : 'bg-red-100 text-red-800 dark:bg-red-900/60 dark:text-red-300'
            }`}
            title={backendOk ? 'Backend reachable on :8765' : 'Backend unreachable'}
          >
            <span
              className={`h-2 w-2 rounded-full ${backendOk ? 'bg-emerald-500' : 'bg-red-500'}`}
              aria-hidden
            />
            {backendOk ? 'API ok' : 'API down'}
          </span>
          <button
            className="btn-secondary !px-2"
            onClick={toggle}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </header>

      <div className="mx-auto flex w-full max-w-[1600px]">
        {/* desktop sidebar */}
        <aside className="sticky top-14 hidden h-[calc(100vh-3.5rem)] w-60 shrink-0 border-r border-slate-200 dark:border-slate-800 lg:block">
          {nav}
        </aside>

        {/* mobile drawer */}
        {open && (
          <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
            <div
              className="absolute inset-0 bg-black/40"
              onClick={() => setOpen(false)}
              aria-hidden
            />
            <div className="absolute left-0 top-14 h-[calc(100vh-3.5rem)] w-64 border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
              {nav}
            </div>
          </div>
        )}

        <main className="min-w-0 flex-1 p-3 sm:p-4 lg:p-6">
          <Outlet />
        </main>
      </div>

      <footer className="border-t border-slate-200 px-4 py-3 text-center text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        Research prototype — not a medical device. Not for clinical use. Seed 20261002 ·
        Regensburg pediatric appendicitis research dataset.
      </footer>
    </div>
  )
}

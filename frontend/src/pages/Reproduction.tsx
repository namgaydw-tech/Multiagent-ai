import { useState } from 'react'
import { Check, Copy, TerminalSquare } from 'lucide-react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { Badge, Card, EmptyState, ErrorBox, Loading, PageHeader } from '../components/states'
import type { ReproCommands } from '../lib/types'

const SECTIONS: { key: keyof Omit<ReproCommands, 'seed' | 'disclaimer'>; title: string; blurb: string }[] = [
  { key: 'train', title: '1 · Train the model (Phase 2)', blurb: 'Preprocessing, split, LightGBM + baselines, calibration, SHAP, metrics.' },
  { key: 'agents_cases', title: '2 · Run agent cases (Phase 3)', blurb: 'Five-agent debiasing engine on example cases with persisted audits.' },
  { key: 'experiments', title: '3 · Run experiments (Phase 4)', blurb: 'Anchor conditions, ablations, then analyze (metrics + figures).' },
  { key: 'tests', title: '4 · Test suite', blurb: 'Everything: preprocessing, models, agents, RAG, bias metrics, API.' },
  { key: 'backend', title: '5 · Start the backend API', blurb: 'FastAPI on port 8765 — serves the real persisted artifacts.' },
  { key: 'frontend', title: '6 · Start the frontend', blurb: 'Vite dev server on port 5199, proxying /api to the backend.' },
]

/** Page 9 — exact reproduction commands + implementation status (IMPLEMENTED vs PLANNED vs VALIDATED). */
export function Reproduction() {
  const repro = useAsync(() => api.repro(), [])
  const finalResults = useAsync(() => api.finalResults(), [])

  const status = (
    finalResults.data?.available
      ? (finalResults.data.data as Record<string, unknown>)?.implementation_status
      : null
  ) as
    | { implemented?: string[]; experimentally_validated_here?: string; not_validated?: string[] }
    | null

  return (
    <div>
      <PageHeader
        title="Reproduction"
        description="Exact commands to regenerate every artifact in this repository. All experiments use seed 20261002."
      />

      {repro.loading && <Loading label="Loading commands…" />}
      {repro.error && <ErrorBox error={repro.error} onRetry={repro.reload} />}

      {repro.data && (
        <div className="space-y-4">
          {SECTIONS.map(({ key, title, blurb }) => (
            <Card key={key} title={title} subtitle={blurb}>
              <CommandBlock lines={repro.data![key] ?? []} />
            </Card>
          ))}

          <Card title="Environment facts" subtitle="Verified in this repository">
            <div className="flex flex-wrap gap-2 text-xs">
              <Badge tone="blue">seed {repro.data.seed}</Badge>
              <Badge>Python 3.13</Badge>
              <Badge>LightGBM 4.7</Badge>
              <Badge>FastAPI + uvicorn</Badge>
              <Badge>Vite + React 18 + Tailwind</Badge>
              <Badge tone={repro.data.disclaimer.includes('not') ? 'amber' : 'slate'}>
                research prototype
              </Badge>
            </div>
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
              {repro.data.disclaimer}
            </p>
          </Card>

          <Card
            title="Implementation status"
            subtitle="Explicitly distinguishing IMPLEMENTED from EXPERIMENTALLY VALIDATED from NOT VALIDATED"
          >
            {finalResults.loading && <Loading label="Loading status…" />}
            {finalResults.error && <ErrorBox error={finalResults.error} onRetry={finalResults.reload} />}
            {!finalResults.data?.available && !finalResults.loading && !finalResults.error && (
              <EmptyState
                title="final_results.json not generated"
                reason={finalResults.data?.reason}
                hint={finalResults.data?.hint ?? 'python scripts/run_phase4.py analyze'}
              />
            )}
            {status && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <div>
                  <p className="mb-1 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
                    <Check className="h-3.5 w-3.5" aria-hidden /> Implemented
                  </p>
                  <ul className="list-inside list-disc space-y-1 text-sm text-slate-600 dark:text-slate-300">
                    {(status.implemented ?? []).map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="mb-1 text-xs font-bold uppercase tracking-wide text-blue-600 dark:text-blue-400">
                    Experimentally validated here
                  </p>
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    {status.experimentally_validated_here ?? 'n/a'}
                  </p>
                </div>
                <div>
                  <p className="mb-1 text-xs font-bold uppercase tracking-wide text-amber-600 dark:text-amber-400">
                    Not validated
                  </p>
                  <ul className="list-inside list-disc space-y-1 text-sm text-slate-600 dark:text-slate-300">
                    {(status.not_validated ?? []).map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}

/** Copy-to-clipboard shell command block. */
function CommandBlock({ lines }: { lines: string[] }) {
  const [copied, setCopied] = useState(false)
  const text = lines.join('\n')

  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  if (lines.length === 0) {
    return <p className="text-sm text-slate-400">no commands recorded</p>
  }

  return (
    <div className="relative">
      <button
        className="btn-secondary absolute right-2 top-2 !px-2 !py-1 text-xs"
        onClick={() => void copy()}
        aria-label="Copy commands"
      >
        {copied ? <Check className="h-3.5 w-3.5" aria-hidden /> : <Copy className="h-3.5 w-3.5" aria-hidden />}
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre className="overflow-x-auto rounded-lg bg-slate-950 p-3 pr-20 text-xs leading-relaxed text-emerald-100">
        {lines.map((line, i) => (
          <div key={i}>
            <span className="select-none text-slate-500">$ </span>
            {line}
          </div>
        ))}
      </pre>
      <p className="mt-1 flex items-center gap-1 text-[11px] text-slate-400">
        <TerminalSquare className="h-3 w-3" aria-hidden />
        run from the repository root (frontend commands include cd frontend)
      </p>
    </div>
  )
}

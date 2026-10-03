import { useState } from 'react'
import { BrainCircuit, ChevronDown, ChevronRight, ShieldAlert } from 'lucide-react'
import { ABLATIONS, ANCHOR_CONDITIONS, api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { num } from '../lib/format'
import { Badge, Card, EmptyState, ErrorBox, Loading, MetricTile, PageHeader } from '../components/states'
import type { AgentRunResponse, StageSummary } from '../lib/types'

/** Page 4 — run the real 8-stage adversarial pipeline and inspect each stage. */
export function AgentTimeline() {
  const cases = useAsync(() => api.cases(500), [])
  const [row, setRow] = useState<number>(469)
  const [ablation, setAblation] = useState<string>('default')
  const [anchor, setAnchor] = useState<string>('')
  const [rag, setRag] = useState(true)
  const [run, setRun] = useState<AgentRunResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  async function execute() {
    setRunning(true)
    setError(null)
    setRun(null)
    try {
      const res = await api.runAgents({
        row_index: row,
        ablation,
        rag_enabled: rag,
        anchor_condition: anchor === '' ? null : anchor,
      })
      setRun(res)
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)))
    } finally {
      setRunning(false)
    }
  }

  const availableRows = cases.data?.rows ?? []
  const rowValid = availableRows.some((r) => r.row_index === row) || row > 0

  return (
    <div>
      <PageHeader
        title="Agent Timeline"
        description="Execute the genuine state machine: data cleansing → independent differential → proponent → opponent → RAG retrieval → watchdog → arbitration → bias audit. Each run persists a full audit under outputs/audits/."
      />

      <Card title="Run configuration">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              Case row
            </span>
            <input
              type="number"
              className="input w-full"
              value={row}
              min={0}
              onChange={(e) => setRow(Number(e.target.value))}
            />
            <span className="mt-1 block text-xs text-slate-400">
              {cases.loading
                ? 'loading cases…'
                : availableRows.length > 0
                  ? `${availableRows.length} test rows available`
                  : 'enter any modelling row index'}
            </span>
          </label>

          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              Ablation policy
            </span>
            <select className="input w-full" value={ablation} onChange={(e) => setAblation(e.target.value)}>
              {ABLATIONS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>

          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              Anchor condition (optional)
            </span>
            <select className="input w-full" value={anchor} onChange={(e) => setAnchor(e.target.value)}>
              <option value="">none (no anchor injected)</option>
              {ANCHOR_CONDITIONS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>

          <label className="flex cursor-pointer items-center gap-2 self-end pb-1 text-sm">
            <input
              type="checkbox"
              checked={rag}
              onChange={(e) => setRag(e.target.checked)}
              className="h-4 w-4 accent-brand-600"
            />
            <span>
              RAG retrieval enabled
              <span className="block text-xs text-slate-400">72 offline knowledge chunks</span>
            </span>
          </label>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button className="btn" onClick={() => void execute()} disabled={running || !rowValid}>
            <BrainCircuit className="h-4 w-4" aria-hidden />
            {running ? 'Running stages…' : 'Run 8-stage pipeline'}
          </button>
          {!rowValid && (
            <span className="text-xs text-red-500">row index must be a positive integer</span>
          )}
        </div>
      </Card>

      {error && (
        <div className="mt-4">
          <ErrorBox error={error} />
        </div>
      )}
      {running && (
        <div className="mt-4">
          <Loading label="Agents are debating — deterministic backend takes ~0.5–2 s…" />
        </div>
      )}

      {!run && !running && !error && (
        <div className="mt-4">
          <EmptyState
            title="No run yet"
            reason="Configure a case and run the pipeline. Everything below is real output from the engine — nothing is simulated for display."
          />
        </div>
      )}

      {run && !running && (
        <>
          <div className="mt-4 grid grid-cols-2 gap-3 xl:grid-cols-4">
            <MetricTile
              label="Model top-1"
              value={String(run.summary.model_top ?? 'n/a')}
              sub={
                typeof run.summary.model_probability === 'number'
                  ? `p=${num(run.summary.model_probability, 3)}`
                  : undefined
              }
            />
            <MetricTile
              label="Final working diagnosis"
              value={String(run.summary.primary_working_diagnosis ?? 'n/a')}
              sub={`ground truth: ${run.ground_truth ?? 'n/a'}`}
              tone={
                run.summary.primary_working_diagnosis === run.ground_truth ? 'good' : 'warn'
              }
            />
            <MetricTile
              label="Multi-agent confidence"
              value={
                typeof run.summary.multi_agent_confidence === 'number'
                  ? num(run.summary.multi_agent_confidence, 3)
                  : 'n/a'
              }
              sub={`watchdog: ${String(run.summary.watchdog_risk ?? 'n/a')}`}
            />
            <MetricTile
              label="Elapsed"
              value={`${num(run.elapsed_s, 2)}s`}
              sub={`${run.stages.length} stages · ${run.ablation}`}
            />
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
            <Badge tone="blue">case {run.case_id}</Badge>
            <Badge tone={run.rag_enabled ? 'green' : 'slate'}>RAG {run.rag_enabled ? 'on' : 'off'}</Badge>
            {run.anchor ? (
              <Badge tone="amber">
                anchor: {String(run.anchor.value ?? 'set')} ({String(run.anchor.source ?? run.anchor.method ?? 'injected')})
              </Badge>
            ) : (
              <Badge>no anchor</Badge>
            )}
            <span className="min-w-0 break-all text-slate-400">{run.audit_path}</span>
          </div>

          {Object.keys(run.summary.errors ?? {}).length > 0 && (
            <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
              <p className="flex items-center gap-1.5 font-semibold">
                <ShieldAlert className="h-4 w-4" aria-hidden /> Stage errors (recorded, not hidden)
              </p>
              <ul className="mt-1 list-inside list-disc">
                {Object.entries(run.summary.errors ?? {}).map(([stage, message]) => (
                  <li key={stage}>{stage}: {message}</li>
                ))}
              </ul>
            </div>
          )}

          <Card title="Stage timeline" subtitle="Visibility badges show what each agent was allowed to see" className="mt-4">
            <ol className="space-y-2">
              {run.stages.map((stage) => (
                <StageRow key={stage.stage} stage={stage} />
              ))}
            </ol>
          </Card>

          <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">{run.disclaimer}</p>
        </>
      )}
    </div>
  )
}

/** One expandable stage entry: badges + full JSON output on demand. */
function StageRow({ stage }: { stage: StageSummary }) {
  const [open, setOpen] = useState(false)
  const v = stage.visibility
  return (
    <li className="rounded-lg border border-slate-200 dark:border-slate-700">
      <button
        className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        {open ? (
          <ChevronDown className="h-4 w-4 shrink-0 text-slate-400" aria-hidden />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0 text-slate-400" aria-hidden />
        )}
        <span className="w-6 shrink-0 text-xs font-bold text-slate-400">{stage.stage}</span>
        <span className="min-w-0 flex-1 truncate text-sm font-medium">{stage.name}</span>        {stage.backend && <Badge tone="slate">{stage.backend}</Badge>}
        <Badge tone={v.sees_prediction ? 'blue' : 'slate'}>
          {v.sees_prediction ? 'sees prediction' : 'blind to prediction'}
        </Badge>
        <Badge tone={v.sees_anchor ? 'amber' : 'slate'}>
          {v.sees_anchor ? 'sees anchor' : 'no anchor'}
        </Badge>
        {v.sees_ground_truth && <Badge tone="red">sees truth</Badge>}
        {stage.latency_s !== null && (
          <span className="hidden w-14 shrink-0 text-right text-xs tabular-nums text-slate-400 sm:block">
            {num(stage.latency_s, 3)}s
          </span>
        )}
      </button>
      {open && (
        <div className="border-t border-slate-100 px-3 py-2 dark:border-slate-800">
          {stage.withheld.length > 0 && (
            <p className="mb-1 text-xs text-slate-500">
              withheld from this stage: {stage.withheld.join(', ')}
            </p>
          )}
          {stage.prompt_hash && (
            <p className="mb-1 text-xs text-slate-400">prompt hash {stage.prompt_hash}</p>
          )}
          <pre className="max-h-72 overflow-auto rounded bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-100">
            {JSON.stringify(stage.output, null, 2)}
          </pre>
        </div>
      )}
    </li>
  )
}

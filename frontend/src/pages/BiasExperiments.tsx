import { useState } from 'react'
import { Play } from 'lucide-react'
import { ANCHOR_CONDITIONS, PROFILES, api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { metricWithCI, num, pct, pValue } from '../lib/format'
import { Badge, Card, EmptyState, ErrorBox, Loading, MetricTile, PageHeader } from '../components/states'
import type { ConditionStats, Headline, MetricRecord, QuickExperimentResponse } from '../lib/types'

const METRIC_KEYS: { key: keyof Headline; label: string; kind: 'rate' | 'raw' }[] = [
  { key: 'CBR', label: 'CBR', kind: 'rate' },
  { key: 'AOR', label: 'AOR', kind: 'rate' },
  { key: 'BCR', label: 'BCR', kind: 'rate' },
  { key: 'HFR', label: 'HFR', kind: 'rate' },
  { key: 'CRR', label: 'CRR', kind: 'rate' },
  { key: 'CMR', label: 'CMR', kind: 'rate' },
  { key: 'final_accuracy', label: 'Accuracy', kind: 'rate' },
]

function cell(headline: Headline | undefined, key: keyof Headline, kind: 'rate' | 'raw'): string {
  const rec = headline?.[key] as MetricRecord | undefined
  if (!rec || rec.value === null || rec.value === undefined) return 'n/a'
  return kind === 'rate' ? pct(rec.value, 1) : metricWithCI(rec)
}

/** Page 5 — Phase 4 anchor-condition experiment: bias metrics + statistics + quick runner. */
export function BiasExperiments() {
  const bias = useAsync(() => api.bias(), [])
  const stats: ConditionStats | null = bias.data?.available
    ? (bias.data.data as unknown as ConditionStats)
    : null

  const anchorExp = stats?.anchor_experiment
  const conditions = anchorExp ? Object.keys(anchorExp.conditions) : []
  const br = stats?.bias_reduction

  return (
    <div>
      <PageHeader
        title="Bias Experiments"
        description="Phase 4 — anchor injection across 7 conditions, 117 test cases × 3 repetitions, with bootstrap CIs, exact McNemar tests and Benjamini–Hochberg correction."
      />

      {bias.loading && <Loading label="Loading bias metrics…" />}
      {bias.error && <ErrorBox error={bias.error} onRetry={bias.reload} />}
      {bias.data && !bias.data.available && (
        <EmptyState
          title="Bias experiment results not generated"
          reason={bias.data.reason}
          hint={bias.data.hint ?? 'python scripts/run_phase4.py anchors && python scripts/run_phase4.py analyze'}
        />
      )}

      {stats && (
        <>
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <MetricTile
              label="CBR — multi-agent"
              value={metricWithCI(br?.cbr_multi)}
              sub={`n=${br?.cbr_multi?.n_den ?? 'n/a'} · incorrect anchors`}
              tone="good"
            />
            <MetricTile
              label="CBR — single agent + model"
              value={metricWithCI(br?.cbr_single_3)}
              sub="profile 3 baseline"
              tone="bad"
            />
            <MetricTile
              label="Bias reduction (paired)"
              value={typeof br?.paired_diff?.diff === 'number' ? pct(br.paired_diff.diff, 1) : 'n/a'}
              sub={`p=${pValue(br?.paired_diff?.p_value)} · n=${br?.paired_diff?.n_pairs ?? 'n/a'}`}
              tone="good"
            />
            <MetricTile
              label="McNemar (final accuracy)"
              value={`b=${br?.mcnemar?.b ?? 'n/a'} / c=${br?.mcnemar?.c ?? 'n/a'}`}
              sub={`p=${pValue(br?.mcnemar?.p_value)}`}
            />
          </div>

          {anchorExp && (
            <Card
              title="Anchor conditions — full five-agent profile"
              subtitle={`profile ${anchorExp.profile} · n_cases=${anchorExp.n_cases} · reps=${anchorExp.reps} · values are point estimates with 95% bootstrap CI`}
              className="mt-4"
            >
              <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
                <table className="table-base min-w-[900px]">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      {conditions.map((c) => (
                        <th key={c} className="text-right" title={c}>
                          <span className="block max-w-[110px] truncate">{c}</span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {METRIC_KEYS.map(({ key, label, kind }) => (
                      <tr key={key}>
                        <td className="font-medium">{label}</td>
                        {conditions.map((c) => (
                          <td key={c} className="whitespace-nowrap text-right tabular-nums">
                            {cell(anchorExp.conditions[c], key, kind)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                CBR is undefined (n/a) for conditions where the denominator is 0 — undefined
                metrics are never rendered as 0.
              </p>
            </Card>
          )}

          <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card title="Single-agent baselines" subtitle="CBR per anchor condition (lower is better)">
              <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
                <table className="table-base min-w-[520px]">
                  <thead>
                    <tr>
                      <th>Profile</th>
                      {ANCHOR_CONDITIONS.map((c) => (
                        <th key={c} className="text-right">
                          <span className="block max-w-[80px] truncate" title={c}>
                            {c.replace('_anchor', '').replace('_incorrect', '.inc')}
                          </span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(stats.single_baselines ?? {}).map(([profile, conds]) => (
                      <tr key={profile}>
                        <td className="whitespace-nowrap font-medium">{profile}</td>
                        {ANCHOR_CONDITIONS.map((c) => (
                          <td key={c} className="text-right tabular-nums">
                            {cell(conds[c], 'CBR', 'rate')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card
              title="Statistical comparisons"
              subtitle="Exact tests + Benjamini–Hochberg FDR correction"
            >
              <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
                <table className="table-base min-w-[460px]">
                  <thead>
                    <tr>
                      <th>Comparison</th>
                      <th>Test</th>
                      <th className="text-right">Δ</th>
                      <th className="text-right">p</th>
                      <th className="text-right">q (BH)</th>
                      <th className="text-right">Rejected</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(stats.statistics?.comparisons ?? []).map((c, i) => (
                      <tr key={`${c.name}-${i}`}>
                        <td className="max-w-[220px] truncate" title={c.name}>{c.name}</td>
                        <td className="text-xs text-slate-500">{c.test}</td>
                        <td className="text-right tabular-nums">
                          {typeof c.diff === 'number' ? num(c.diff, 3) : 'n/a'}
                        </td>
                        <td className="text-right tabular-nums">{pValue(c.p_value ?? null)}</td>
                        <td className="text-right tabular-nums">{pValue(c.q_value ?? null)}</td>
                        <td className="text-right">
                          <Badge tone={c.rejected_bh ? 'green' : 'slate'}>
                            {c.rejected_bh ? 'yes' : 'no'}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                    {(stats.statistics?.comparisons ?? []).length === 0 && (
                      <tr>
                        <td colSpan={6} className="py-3 text-center text-slate-400">
                          no comparisons recorded
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                BH α={stats.statistics?.benjamini_hochberg?.alpha ?? 'n/a'},{' '}
                {stats.statistics?.benjamini_hochberg?.n_tests ?? 0} tests.
              </p>
            </Card>
          </div>

          <Card
            title="Experiment figures"
            subtitle="Rendered from outputs/figures (Phase 4 analyze step)"
            className="mt-4"
          >
            <BiasFigures />
          </Card>

          <QuickRunner />
        </>
      )}
    </div>
  )
}

function BiasFigures() {
  const health = useAsync(() => api.health(), [])
  const available = new Set(health.data?.figures ?? [])
  const figs = ['fig_anchor_cbr_aor.png', 'fig_anchor_outcomes.png', 'fig_bias_reduction.png']
  if (health.loading) return <Loading label="Checking figures…" />
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {figs.map((name) => (
        <figure key={name} className="rounded-lg border border-slate-200 p-2 dark:border-slate-700">
          <figcaption className="mb-1.5 text-xs font-medium text-slate-500 dark:text-slate-400">
            {name}
          </figcaption>
          {available.has(name) ? (
            <img src={api.figureUrl(name)} alt={name} loading="lazy" className="w-full rounded bg-white dark:bg-slate-950" />
          ) : (
            <div className="flex h-32 items-center justify-center rounded bg-slate-50 p-2 text-center text-xs text-slate-400 dark:bg-slate-800/50">
              not generated — run python scripts/run_phase4.py analyze
            </div>
          )}
        </figure>
      ))}
    </div>
  )
}

/** Small synchronous experiment runner (N ≤ 15) for interactive exploration. */
function QuickRunner() {
  const [conditions, setConditions] = useState<string[]>(['control', 'incorrect_anchor'])
  const [profiles, setProfiles] = useState<string[]>(['9_full_with_rag'])
  const [nCases, setNCases] = useState(5)
  const [result, setResult] = useState<QuickExperimentResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  function toggle(list: string[], set: (v: string[]) => void, value: string) {
    set(list.includes(value) ? list.filter((v) => v !== value) : [...list, value])
  }

  async function run() {
    if (conditions.length === 0 || profiles.length === 0) {
      setError(new Error('Pick at least one condition and one profile'))
      return
    }
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.quickExperiment({ conditions, profiles, n_cases: nCases })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)))
    } finally {
      setRunning(false)
    }
  }

  return (
    <Card
      title="Quick experiment runner"
      subtitle="Runs real cases through the real framework synchronously (N ≤ 15, repetition 0)"
      className="mt-4"
    >
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <fieldset>
          <legend className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">
            Anchor conditions
          </legend>
          <div className="flex flex-wrap gap-1.5">
            {ANCHOR_CONDITIONS.map((c) => (
              <button
                key={c}
                className={`rounded-full border px-2.5 py-1 text-xs transition ${
                  conditions.includes(c)
                    ? 'border-brand-600 bg-brand-600 text-white'
                    : 'border-slate-300 text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800'
                }`}
                onClick={() => toggle(conditions, setConditions, c)}
                aria-pressed={conditions.includes(c)}
              >
                {c}
              </button>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">
            Profiles
          </legend>
          <div className="flex flex-wrap gap-1.5">
            {PROFILES.map((p) => (
              <button
                key={p}
                className={`rounded-full border px-2.5 py-1 text-xs transition ${
                  profiles.includes(p)
                    ? 'border-brand-600 bg-brand-600 text-white'
                    : 'border-slate-300 text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800'
                }`}
                onClick={() => toggle(profiles, setProfiles, p)}
                aria-pressed={profiles.includes(p)}
              >
                {p}
              </button>
            ))}
          </div>
        </fieldset>

        <div className="flex items-end gap-3">
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              Cases (max 15)
            </span>
            <input
              type="number"
              className="input w-24"
              min={1}
              max={15}
              value={nCases}
              onChange={(e) => setNCases(Math.max(1, Math.min(15, Number(e.target.value) || 1)))}
            />
          </label>
          <button className="btn" onClick={() => void run()} disabled={running}>
            <Play className="h-4 w-4" aria-hidden />
            {running ? 'Running…' : 'Run experiment'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-3">
          <ErrorBox error={error} />
        </div>
      )}
      {running && (
        <div className="mt-3">
          <Loading label="Executing cases — up to ~30 s for 15 cases × conditions × profiles…" />
        </div>
      )}

      {result && (
        <div className="mt-4 space-y-3">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge tone="blue">{result.n_cases} cases</Badge>
            <Badge>{result.elapsed_s}s</Badge>
            <Badge tone="amber">{result.rows.length} case runs</Badge>
          </div>
          <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
            <table className="table-base min-w-[560px]">
              <thead>
                <tr>
                  <th>profile | condition</th>
                  <th className="text-right">CBR</th>
                  <th className="text-right">HFR</th>
                  <th className="text-right">Accuracy</th>
                  <th className="text-right">n_rows</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.headline).map(([key, h]) => (
                  <tr key={key}>
                    <td className="font-mono text-xs">{key}</td>
                    <td className="text-right tabular-nums">{cell(h, 'CBR', 'rate')}</td>
                    <td className="text-right tabular-nums">{cell(h, 'HFR', 'rate')}</td>
                    <td className="text-right tabular-nums">{cell(h, 'final_accuracy', 'rate')}</td>
                    <td className="text-right tabular-nums">
                      {(h.n_rows as number | undefined) ?? 'n/a'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">{result.note}</p>
          <p className="text-xs text-slate-400">{result.disclaimer}</p>
        </div>
      )}
    </Card>
  )
}

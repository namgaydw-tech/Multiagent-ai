import { useMemo, useState } from 'react'
import { Trophy } from 'lucide-react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { Badge, Card, EmptyState, ErrorBox, Loading, PageHeader } from '../components/states'
import type { ScorecardRow } from '../lib/api'

const COLUMNS: { key: keyof ScorecardRow; label: string }[] = [
  { key: 'Macro_F0.5', label: 'Macro F0.5' },
  { key: 'positive_F0.5', label: 'pos F0.5' },
  { key: 'accuracy', label: 'Acc' },
  { key: 'balanced_accuracy', label: 'Bal Acc' },
  { key: 'precision', label: 'Prec' },
  { key: 'sensitivity', label: 'Sens' },
  { key: 'specificity', label: 'Spec' },
  { key: 'NPV', label: 'NPV' },
  { key: 'F1', label: 'F1' },
  { key: 'F2', label: 'F2' },
  { key: 'MCC', label: 'MCC' },
  { key: 'AUROC', label: 'AUROC' },
  { key: 'AUPRC', label: 'AUPRC' },
  { key: 'Brier', label: 'Brier' },
  { key: 'ECE', label: 'ECE' },
]

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined) return 'n/a'
  return typeof v === 'number' ? v.toFixed(3) : String(v)
}

/** Page 11 — Phase 5 cross-dataset benchmark: scorecard, gaps, paired statistics. */
export function CrossDatasetBenchmark() {
  const scorecard = useAsync(() => api.phase5Scorecard(), [])
  const cross = useAsync(() => api.phase5CrossDataset(), [])
  const stats = useAsync(() => api.phase5Statistics(), [])
  const [dataset, setDataset] = useState<string>('ALL')

  const rows: ScorecardRow[] = scorecard.data?.data?.rows ?? []
  const datasetIds = useMemo(() => Array.from(new Set(rows.map((r) => r.dataset))), [rows])
  const shown = dataset === 'ALL' ? rows : rows.filter((r) => r.dataset === dataset)

  const crossData = (cross.data?.available ? cross.data.data : null) as
    | { datasets?: Record<string, unknown>[] }
    | null
  const statsData = (stats.data?.available ? stats.data.data : null) as
    | { datasets?: Record<string, unknown> }
    | null

  return (
    <div>
      <PageHeader
        title="Cross-Dataset Benchmark"
        description="Model scorecard across separate pediatric experiments — datasets are never concatenated. Winners are selected on VALIDATION only (Macro F0.5 subject to the sensitivity floor), then the sealed test is opened once."
      />

      <Card
        title="Scorecard"
        subtitle={
          scorecard.data?.data
            ? `${scorecard.data.data.n_rows} rows · ${scorecard.data.data.winner_selection}`
            : 'model_scorecard.csv / .json'
        }
      >
        {scorecard.loading && <Loading label="Loading scorecard…" />}
        {scorecard.error && <ErrorBox error={scorecard.error} onRetry={scorecard.reload} />}
        {!scorecard.data?.available && !scorecard.loading && !scorecard.error && (
          <EmptyState
            title="model_scorecard.json not generated"
            reason={scorecard.data?.reason}
            hint={scorecard.data?.hint ?? 'python scripts/run_phase5.py analyze'}
          />
        )}
        {rows.length > 0 && (
          <>
            <div className="mb-3 flex flex-wrap gap-2">
              {['ALL', ...datasetIds].map((d) => (
                <button
                  key={d}
                  onClick={() => setDataset(d)}
                  className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                    dataset === d
                      ? 'bg-brand-600 text-white'
                      : 'bg-slate-200 text-slate-700 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300'
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 uppercase tracking-wide text-slate-500 dark:border-slate-700">
                    <th className="py-2 pr-2">Algorithm</th>
                    <th className="py-2 pr-2">Family</th>
                    <th className="py-2 pr-2">Status</th>
                    <th className="py-2 pr-2">Thr</th>
                    {COLUMNS.map((c) => (
                      <th key={String(c.key)} className="py-2 pr-2">
                        {c.label}
                      </th>
                    ))}
                    <th className="py-2 pr-2">Calib</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((r) => (
                    <tr
                      key={`${r.dataset}-${r.algorithm}`}
                      className={`border-b border-slate-100 dark:border-slate-800 ${
                        r.winner ? 'bg-amber-50 dark:bg-amber-900/20' : ''
                      }`}
                    >
                      <td className="py-1.5 pr-2 font-medium">
                        {r.winner && (
                          <Trophy
                            className="mr-1 inline h-3 w-3 text-amber-500"
                            aria-hidden
                          />
                        )}
                        {r.algorithm}
                      </td>
                      <td className="py-1.5 pr-2 text-slate-500">{r.model_family}</td>
                      <td className="py-1.5 pr-2">
                        <Badge
                          tone={
                            r.status === 'EVALUATED'
                              ? 'green'
                              : r.status === 'UNAVAILABLE'
                                ? 'amber'
                                : 'slate'
                          }
                        >
                          {r.status}
                        </Badge>
                      </td>
                      <td className="py-1.5 pr-2 tabular-nums">{fmt(r.threshold)}</td>
                      {COLUMNS.map((c) => (
                        <td key={String(c.key)} className="py-1.5 pr-2 tabular-nums">
                          {fmt(r[c.key] as number | null)}
                        </td>
                      ))}
                      <td className="py-1.5 pr-2 text-slate-500">
                        {r.calibration_method ?? 'n/a'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
              Unavailable metrics show n/a — never 0 (e.g. hard voting has no probabilities,
              so AUROC is null). Thresholds locked on validation before test.
            </p>
          </>
        )}
      </Card>

      <Card title="Cross-dataset view" subtitle="Winners, generalization gaps, blocker status">
        {cross.loading && <Loading label="Loading cross-dataset summary…" />}
        {cross.error && <ErrorBox error={cross.error} onRetry={cross.reload} />}
        {!cross.data?.available && !cross.loading && !cross.error && (
          <EmptyState
            title="cross_dataset_summary.json not generated"
            reason={cross.data?.reason}
            hint={cross.data?.hint ?? 'python scripts/run_phase5.py analyze'}
          />
        )}
        {crossData?.datasets && (
          <div className="grid gap-3 md:grid-cols-2">
            {crossData.datasets.map((d) => {
              const rec = d as Record<string, unknown>
              return (
                <div
                  key={String(rec.dataset)}
                  className="rounded-lg border border-slate-200 p-3 dark:border-slate-700"
                >
                  <p className="font-mono text-xs font-semibold">{String(rec.dataset)}</p>
                  <div className="mt-2 flex flex-wrap gap-2 text-xs">
                    <Badge tone={String(rec.blocker_status) === 'PASS' ? 'green' : 'amber'}>
                      {String(rec.blocker_status)}
                    </Badge>
                    <Badge tone="blue">winner: {String(rec.winner)}</Badge>
                    <Badge>val {fmt(rec.winner_val_macro_f0_5 as number)}</Badge>
                    <Badge>test {fmt(rec.winner_test_macro_f0_5 as number)}</Badge>
                    <Badge tone="amber">
                      gap {fmt(rec.generalization_gap_val_minus_test as number)}
                    </Badge>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </Card>

      <Card
        title="Statistical comparison"
        subtitle="Paired bootstrap Δ Macro F0.5, exact McNemar, DeLong AUROC CI, BH correction — post-hoc on the sealed test"
      >
        {stats.loading && <Loading label="Loading statistics…" />}
        {stats.error && <ErrorBox error={stats.error} onRetry={stats.reload} />}
        {!stats.data?.available && !stats.loading && !stats.error && (
          <EmptyState
            title="statistical_comparison.json not generated"
            reason={stats.data?.reason}
            hint={stats.data?.hint ?? 'python scripts/run_phase5.py analyze'}
          />
        )}
        {statsData?.datasets &&
          Object.entries(statsData.datasets).map(([ds, entry]) => {
            const e = entry as {
              winner?: string
              n_test?: number
              benjamini_hochberg?: { n_tests?: number; rejected_q_lt_0_05?: string[] }
              comparisons?: {
                competitor: string
                paired_bootstrap_delta_macro_f0_5: {
                  delta_macro_f0_5: number
                  ci95: number[]
                  p_value: number
                }
                mcnemar: { b: number; c: number; p_value: number | null }
              }[]
            }
            return (
              <div key={ds} className="mb-4">
                <p className="mb-1 text-sm font-semibold">
                  {ds} — winner {e.winner} on n={e.n_test} sealed-test cases
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 uppercase text-slate-500 dark:border-slate-700">
                        <th className="py-1 pr-2">Competitor</th>
                        <th className="py-1 pr-2">Δ Macro F0.5</th>
                        <th className="py-1 pr-2">95% CI</th>
                        <th className="py-1 pr-2">p</th>
                        <th className="py-1 pr-2">McNemar b/c</th>
                        <th className="py-1 pr-2">q (BH)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(e.comparisons ?? []).slice(0, 8).map((c) => (
                        <tr
                          key={c.competitor}
                          className="border-b border-slate-100 dark:border-slate-800"
                        >
                          <td className="py-1 pr-2">{c.competitor}</td>
                          <td className="py-1 pr-2 tabular-nums">
                            {fmt(c.paired_bootstrap_delta_macro_f0_5.delta_macro_f0_5)}
                          </td>
                          <td className="py-1 pr-2 tabular-nums">
                            [{fmt(c.paired_bootstrap_delta_macro_f0_5.ci95?.[0])},{' '}
                            {fmt(c.paired_bootstrap_delta_macro_f0_5.ci95?.[1])}]
                          </td>
                          <td className="py-1 pr-2 tabular-nums">
                            {c.paired_bootstrap_delta_macro_f0_5.p_value?.toFixed(4)}
                          </td>
                          <td className="py-1 pr-2 tabular-nums">
                            {c.mcnemar?.b}/{c.mcnemar?.c}
                          </td>
                          <td className="py-1 pr-2 text-slate-500">
                            {e.benjamini_hochberg?.rejected_q_lt_0_05?.includes(c.competitor)
                              ? 'q<0.05'
                              : 'n.s.'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })}
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Tiny point differences without CI separation are not claimed as meaningful. Test
          labels were used only for this post-hoc inference — never for selection.
        </p>
      </Card>
    </div>
  )
}

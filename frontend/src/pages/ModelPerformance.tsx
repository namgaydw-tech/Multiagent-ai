import { AlertTriangle } from 'lucide-react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { num, pct, shortHash } from '../lib/format'
import { Badge, Card, EmptyState, ErrorBox, Loading, PageHeader } from '../components/states'

/** One model entry of outputs/metrics/model_comparison.json. */
interface ModelEntry {
  test_metrics: Record<string, number | null | Record<string, number>>
  bootstrap_ci: Record<string, { point?: number; lo: number | null; hi: number | null; n_boot?: number } | undefined>
  calibrator?: string
  threshold?: number
  val_auprc?: number
  best_params?: Record<string, unknown>
}

interface ComparisonPayload {
  generated_utc?: string
  note?: string
  split_counts?: Record<string, { n: number; n_positive: number; n_negative: number }>
  split_strategy?: string
  config_hash?: string
  models: Record<string, ModelEntry>
  false_negative_review?: {
    n_false_negatives?: number
    false_negative_rate?: number
    cases?: Record<string, unknown>[]
  }
  shap?: { status?: string; n_rows?: number; global_top?: { feature: string; mean_abs_shap: number }[] }
}

const METRIC_COLS: { key: string; label: string; pct?: boolean; digits?: number }[] = [
  { key: 'auroc', label: 'AUROC', digits: 3 },
  { key: 'auprc', label: 'AUPRC', digits: 3 },
  { key: 'sensitivity', label: 'Sens', pct: true },
  { key: 'specificity', label: 'Spec', pct: true },
  { key: 'ppv', label: 'PPV', pct: true },
  { key: 'npv', label: 'NPV', pct: true },
  { key: 'f2', label: 'F2', digits: 3 },
  { key: 'brier', label: 'Brier', digits: 3 },
  { key: 'ece', label: 'ECE', digits: 3 },
]

const FIGURES = [
  { name: 'roc_curve.png', title: 'ROC curve' },
  { name: 'pr_curve.png', title: 'PR curve' },
  { name: 'confusion_matrix.png', title: 'Confusion matrix' },
  { name: 'calibration_curve.png', title: 'Calibration curve' },
  { name: 'shap_summary.png', title: 'SHAP summary' },
]

function fmt(metric: number | null | undefined, pctFlag: boolean, digits = 3): string {
  if (metric === null || metric === undefined || Number.isNaN(metric)) return 'n/a'
  return pctFlag ? pct(metric, 1) : num(metric, digits)
}

/** Page 2 — Phase 2 model comparison: metrics + 2000-bootstrap CIs + figures. */
export function ModelPerformance() {
  const summary = useAsync(() => api.summary(), [])
  const payload = summary.data?.available
    ? (summary.data.data as unknown as ComparisonPayload)
    : null

  return (
    <div>
      <PageHeader
        title="Model Performance"
        description="Phase 2 — LightGBM plus baselines on the Regensburg test partition (n=117). All numbers come from outputs/metrics/model_comparison.json; CIs are 2000-bootstrap percentile intervals."
      />

      {summary.loading && <Loading label="Loading model comparison…" />}
      {summary.error && <ErrorBox error={summary.error} onRetry={summary.reload} />}
      {summary.data && !summary.data.available && (
        <EmptyState
          title="Model comparison not generated yet"
          reason={summary.data.reason}
          hint={summary.data.hint ?? 'python scripts/run_phase2.py'}
        />
      )}

      {payload && (
        <>
          <Card
            title="Test-set metrics (117 cases)"
            subtitle={payload.note}
            actions={
              <div className="flex flex-wrap gap-1.5">
                <Badge tone="blue">seed 20261002</Badge>
                <Badge tone="green">config {shortHash(payload.config_hash)}</Badge>
              </div>
            }
          >
            <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
              <table className="table-base min-w-[900px]">
                <thead>
                  <tr>
                    <th>Model</th>
                    {METRIC_COLS.map((c) => (
                      <th key={c.key} className="text-right">
                        {c.label}
                      </th>
                    ))}
                    <th className="text-right">CM (TN/FP/FN/TP)</th>
                    <th>Calibration</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(payload.models).map(([name, entry]) => {
                    const cm = entry.test_metrics.confusion_matrix as
                      | Record<string, number>
                      | undefined
                    const isPrimary = name === 'lightgbm_appendicitis'
                    return (
                      <tr key={name} className={isPrimary ? 'bg-brand-50/60 dark:bg-brand-600/10' : ''}>
                        <td className="whitespace-nowrap font-medium">
                          {name}
                          {isPrimary && (
                            <Badge tone="blue">
                              <span className="ml-1">primary</span>
                            </Badge>
                          )}
                        </td>
                        {METRIC_COLS.map((c) => {
                          const value = entry.test_metrics[c.key] as number | null | undefined
                          const ci = entry.bootstrap_ci?.[c.key]
                          return (
                            <td key={c.key} className="whitespace-nowrap text-right tabular-nums">
                              {fmt(value, !!c.pct, c.digits)}
                              {ci && ci.lo !== null && ci.hi !== null && (
                                <span className="block text-[11px] text-slate-400 dark:text-slate-500">
                                  [{fmt(ci.lo, !!c.pct, c.digits)}, {fmt(ci.hi, !!c.pct, c.digits)}]
                                </span>
                              )}
                            </td>
                          )
                        })}
                        <td className="whitespace-nowrap text-right tabular-nums">
                          {cm
                            ? `${cm.tn}/${cm.fp}/${cm.fn}/${cm.tp}`
                            : 'n/a'}
                        </td>
                        <td className="whitespace-nowrap text-xs">
                          {entry.calibrator ?? 'n/a'}
                          {entry.threshold !== undefined && ` @ ${entry.threshold}`}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
            <Card title="Patient-level split" subtitle={payload.split_strategy}>
              <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
                <table className="table-base">
                  <thead>
                    <tr>
                      <th>Partition</th>
                      <th className="text-right">n</th>
                      <th className="text-right">positive</th>
                      <th className="text-right">negative</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(payload.split_counts ?? {}).map(([name, counts]) => (
                      <tr key={name}>
                        <td className="font-medium">{name}</td>
                        <td className="text-right tabular-nums">{counts.n}</td>
                        <td className="text-right tabular-nums">{counts.n_positive}</td>
                        <td className="text-right tabular-nums">{counts.n_negative}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card
              title="False-negative review"
              subtitle="Every false negative on the test partition (safety-critical errors)"
            >
              {payload.false_negative_review ? (
                <div className="space-y-2 text-sm">
                  <div className="flex flex-wrap gap-2">
                    <Badge tone="red">
                      {payload.false_negative_review.n_false_negatives ?? 0} FNs
                    </Badge>
                    <Badge>
                      FNR {pct(payload.false_negative_review.false_negative_rate, 1)}
                    </Badge>
                  </div>
                  <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
                    <table className="table-base min-w-[420px]">
                      <thead>
                        <tr>
                          <th>row</th>
                          <th className="text-right">p(cal)</th>
                          <th className="text-right">threshold</th>
                          <th>key features</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(payload.false_negative_review.cases ?? []).map((c) => (
                          <tr key={String(c.row_index)}>
                            <td className="tabular-nums">{String(c.row_index)}</td>
                            <td className="text-right tabular-nums">
                              {num(c.predicted_probability as number, 3)}
                            </td>
                            <td className="text-right tabular-nums">
                              {num(c.threshold as number, 2)}
                            </td>
                            <td className="max-w-[240px] truncate text-xs text-slate-500">
                              {['WBC_Count', 'CRP', 'Appendix_Diameter', 'Alvarado_Score']
                                .filter((k) => c[k] !== undefined && c[k] !== null)
                                .map((k) => `${k}=${String(c[k])}`)
                                .join(' · ') || '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="flex items-start gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" aria-hidden />
                    False negatives are reviewed explicitly, never hidden — see
                    docs/MODEL_CARD.md for the safety discussion.
                  </p>
                </div>
              ) : (
                <EmptyState title="No false-negative review recorded" />
              )}
            </Card>
          </div>

          <Card title="Generated figures" subtitle="Rendered from outputs/figures — missing plots are never fabricated" className="mt-4">
            <FigureGrid />
          </Card>

          {payload.shap?.global_top && payload.shap.global_top.length > 0 && (
            <Card title="Global SHAP importance (top 10)" subtitle={`status: ${payload.shap.status ?? 'ok'} · ${payload.shap.n_rows ?? 0} rows`} className="mt-4">
              <ul className="space-y-1.5">
                {payload.shap.global_top.slice(0, 10).map((f, i) => {
                  const max = payload.shap!.global_top![0].mean_abs_shap || 1
                  return (
                    <li key={f.feature} className="flex items-center gap-2 text-sm">
                      <span className="w-5 shrink-0 text-right text-xs text-slate-400">{i + 1}</span>
                      <span className="w-32 shrink-0 truncate sm:w-64" title={f.feature}>{f.feature}</span>
                      <span className="h-2.5 min-w-0 flex-1 overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
                        <span
                          className="block h-full rounded bg-brand-500"
                          style={{ width: `${Math.max(2, (f.mean_abs_shap / max) * 100)}%` }}
                        />
                      </span>
                      <span className="w-14 shrink-0 text-right text-xs tabular-nums text-slate-500 sm:w-20">
                        {num(f.mean_abs_shap, 4)}
                      </span>
                    </li>
                  )
                })}
              </ul>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                Features contributed to model predictions; SHAP values do not imply causation
                or a clinical diagnosis.
              </p>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

/** Grid of the five canonical Phase 2 figures; each shows a hint when absent. */
function FigureGrid() {
  const health = useAsync(() => api.health(), [])
  const available = new Set(health.data?.figures ?? [])
  if (health.loading) return <Loading label="Checking figures…" />
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {FIGURES.map(({ name, title }) => (
        <figure key={name} className="rounded-lg border border-slate-200 p-2 dark:border-slate-700">
          <figcaption className="mb-1.5 text-xs font-medium text-slate-500 dark:text-slate-400">
            {title}
          </figcaption>
          {available.has(name) ? (
            <img
              src={api.figureUrl(name)}
              alt={title}
              loading="lazy"
              className="w-full rounded bg-white dark:bg-slate-950"
            />
          ) : (
            <div className="flex h-36 items-center justify-center rounded bg-slate-50 text-center text-xs text-slate-400 dark:bg-slate-800/50">
              {name} not generated — run python scripts/run_phase2.py
            </div>
          )}
        </figure>
      ))}
    </div>
  )
}

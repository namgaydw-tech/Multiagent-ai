import { useState } from 'react'
import { Activity, Search } from 'lucide-react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { num, shortHash } from '../lib/format'
import { Card, EmptyState, ErrorBox, Loading, MetricTile, PageHeader } from '../components/states'
import type { PredictResponse } from '../lib/types'

/** Page 3 — browse test cases, run the real model + calibrator + SHAP on one row. */
export function CaseExplorer() {
  const cases = useAsync(() => api.cases(500), [])
  const [selected, setSelected] = useState<number | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<Error | null>(null)
  const [record, setRecord] = useState<Record<string, string | number | null> | null>(null)
  const [filter, setFilter] = useState('')

  async function run(row: number) {
    setSelected(row)
    setRunning(true)
    setRunError(null)
    setResult(null)
    setRecord(null)
    try {
      const [pred, rec] = await Promise.all([api.predict(row), api.caseRecord(row)])
      setResult(pred)
      setRecord(rec.fields)
    } catch (err) {
      setRunError(err instanceof Error ? err : new Error(String(err)))
    } finally {
      setRunning(false)
    }
  }

  const rows = (cases.data?.rows ?? []).filter((r) =>
    filter === '' ? true : String(r.row_index).includes(filter),
  )

  return (
    <div>
      <PageHeader
        title="Case Explorer"
        description="Every test-partition case with its persisted Phase 2 prediction. Running a case executes the real LightGBM model, isotonic calibrator, SHAP explainer and uncertainty estimator."
      />

      {cases.loading && <Loading label="Loading cases…" />}
      {cases.error && <ErrorBox error={cases.error} onRetry={cases.reload} />}
      {cases.data && !cases.data.available && (
        <EmptyState
          title="Predictions not generated"
          reason={cases.data.reason}
          hint={cases.data.hint ?? 'python scripts/run_phase2.py'}
        />
      )}

      {cases.data?.available && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
          <Card
            title={`Test cases (${cases.data.n})`}
            subtitle="Click a row to run the model"
            actions={
              <div className="relative">
                <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" aria-hidden />
                <input
                  className="input w-28 !pl-7 !py-1 text-xs"
                  placeholder="row #"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  aria-label="Filter cases by row index"
                />
              </div>
            }
          >
            <div className="max-h-[560px] overflow-y-auto">
              <table className="table-base">
                <thead className="sticky top-0 bg-white dark:bg-slate-900">
                  <tr>
                    <th>row</th>
                    <th>truth</th>
                    <th className="text-right">p(cal)</th>
                    <th>pred</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const correct = (r.y_true === 1) === (r.pred_class === 'appendicitis')
                    return (
                      <tr
                        key={r.row_index}
                        onClick={() => void run(r.row_index)}
                        className={`cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/60 ${
                          selected === r.row_index ? 'bg-brand-50 dark:bg-brand-600/15' : ''
                        }`}
                      >
                        <td className="tabular-nums">{r.row_index}</td>
                        <td>{r.y_true === 1 ? 'appendicitis' : 'no app.'}</td>
                        <td className="text-right tabular-nums">{num(r.p_calibrated, 3)}</td>
                        <td>
                          <span className={correct ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}>
                            {r.pred_class === 'appendicitis' ? 'app.' : 'no app.'}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={4} className="py-4 text-center text-slate-400">
                        no rows match “{filter}”
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          <div className="space-y-4">
            {selected === null && !running && (
              <EmptyState
                title="Select a case"
                reason="The right panel shows probabilities, SHAP contributors, uncertainty and the raw research record."
              />
            )}
            {running && <Loading label={`Running model on row ${selected}…`} />}
            {runError && <ErrorBox error={runError} />}

            {result && !running && (
              <>
                <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
                  <MetricTile
                    label="Calibrated probability"
                    value={num(result.model_output.calibrated_probability, 3)}
                    sub={`raw ${num(result.model_output.class_probabilities?.['appendicitis'] ?? null, 3)}`}
                  />
                  <MetricTile
                    label="Prediction"
                    value={result.model_output.predicted_class}
                    sub={`threshold applied inside pipeline`}
                    tone={result.model_output.predicted_class === 'appendicitis' ? 'warn' : 'default'}
                  />
                  <MetricTile
                    label="Uncertainty"
                    value={result.uncertainty.uncertainty_level}
                    sub={`H=${num(result.uncertainty.predictive_entropy, 3)} · margin ${num(result.uncertainty.margin, 3)}`}
                    tone={result.uncertainty.uncertainty_level === 'LOW' ? 'good' : result.uncertainty.uncertainty_level === 'HIGH' ? 'bad' : 'warn'}
                  />
                  <MetricTile
                    label="Model"
                    value={shortHash(result.model_output.config_hash)}
                    sub={result.model_output.model_version}
                  />
                </div>

                <Card
                  title="SHAP contributors"
                  subtitle={result.shap_explain.wording_rule}
                >
                  <ul className="space-y-1.5">
                    {result.shap_explain.top_contributors.map((c) => {
                      const width = Math.min(100, (c.contribution / (result.shap_explain.top_contributors[0]?.contribution || 1)) * 100)
                      return (
                        <li key={c.feature} className="flex items-center gap-2 text-sm">
                          <span className="w-24 shrink-0 truncate sm:w-52" title={c.feature}>
                            {c.feature}
                          </span>
                          <span className="hidden w-28 shrink-0 truncate text-xs text-slate-400 sm:block" title={String(c.value ?? '—')}>
                            {c.value === null || c.value === undefined ? 'missing' : String(c.value)}
                          </span>
                          <span className="relative h-3 min-w-0 flex-1 rounded bg-slate-100 dark:bg-slate-800">
                            <span
                              className={`absolute top-0 h-3 rounded ${c.direction === 'positive' ? 'left-1/2 bg-amber-500' : 'right-1/2 bg-sky-500'}`}
                              style={{ width: `${width / 2}%` }}
                              aria-hidden
                            />
                            <span className="absolute left-1/2 top-0 h-3 w-px bg-slate-300 dark:bg-slate-600" aria-hidden />
                          </span>
                          <span className="w-14 shrink-0 text-right text-xs tabular-nums text-slate-500">
                            {num(c.contribution, 4)}
                          </span>
                        </li>
                      )
                    })}
                  </ul>
                  <p className="mt-2 flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                    <Activity className="h-3.5 w-3.5 shrink-0" aria-hidden />
                    Contribution sign: amber = toward “appendicitis”, sky = toward “no
                    appendicitis”. Features contributed to this prediction — no causal claim.
                  </p>
                </Card>

                <Card
                  title="Raw research record"
                  subtitle="Diagnosis / outcome columns stripped for display (information-isolation rule)"
                >
                  {record ? (
                    <div className="grid grid-cols-1 gap-x-4 gap-y-1 text-xs sm:grid-cols-2 xl:grid-cols-3">
                      {Object.entries(record).map(([k, v]) => (
                        <div key={k} className="flex justify-between gap-2 border-b border-slate-100 py-0.5 dark:border-slate-800">
                          <span className="truncate text-slate-500">{k}</span>
                          <span className="truncate font-medium" title={v === null ? 'missing' : String(v)}>
                            {v === null ? <span className="text-slate-400">missing</span> : String(v)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <Loading label="Loading record…" />
                  )}
                </Card>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

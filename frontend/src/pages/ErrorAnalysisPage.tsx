import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { num } from '../lib/format'
import { Badge, Card, EmptyState, ErrorBox, Loading, PageHeader } from '../components/states'

/** Page 7 — Phase 4 error taxonomy: tag counts, rates, FN/FP breakdown, per-case rows. */
export function ErrorAnalysisPage() {
  const analysis = useAsync(() => api.errorAnalysis(), [])
  const payload = analysis.data

  const tagData = Object.entries(payload?.data?.tag_counts ?? {})
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count)

  return (
    <div>
      <PageHeader
        title="Error Analysis"
        description="Where the system fails: explicit error taxonomy over the 117 test cases — model errors, system errors, anchor outcomes and harmful flips."
      />

      {analysis.loading && <Loading label="Loading error analysis…" />}
      {analysis.error && <ErrorBox error={analysis.error} onRetry={analysis.reload} />}
      {payload && payload.available === false && (
        <EmptyState
          title="Error analysis not generated"
          reason={payload.reason}
          hint={payload.hint ?? 'python scripts/run_phase4.py analyze'}
        />
      )}

      {payload?.data && (
        <>
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <Card title="Rates">
              <dl className="space-y-1.5 text-sm">
                {Object.entries(payload.data.rates).map(([k, v]) => (
                  <div key={k} className="flex items-baseline justify-between gap-2">
                    <dt className="min-w-0 break-all text-slate-500 dark:text-slate-400">{k}</dt>
                    <dd className="shrink-0 font-semibold tabular-nums">
                      {v === null ? 'n/a' : num(v, 4)}
                    </dd>
                  </div>
                ))}
              </dl>
            </Card>
            <Card title="False negatives">
              <dl className="space-y-1.5 text-sm">
                {Object.entries(payload.data.fn_breakdown).map(([k, v]) => (
                  <div key={k} className="flex items-baseline justify-between gap-2">
                    <dt className="min-w-0 break-all text-slate-500 dark:text-slate-400">{k}</dt>
                    <dd className="shrink-0 font-semibold tabular-nums">{v}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-2 text-xs text-slate-500">
                All model FNs persist into the system output — the framework does not
                recover misses on this dataset (disclosed honestly).
              </p>
            </Card>
            <Card title="False positives">
              <dl className="space-y-1.5 text-sm">
                {Object.entries(payload.data.fp_breakdown).map(([k, v]) => (
                  <div key={k} className="flex items-baseline justify-between gap-2">
                    <dt className="min-w-0 break-all text-slate-500 dark:text-slate-400">{k}</dt>
                    <dd className="shrink-0 font-semibold tabular-nums">{v}</dd>
                  </div>
                ))}
              </dl>
            </Card>
            <Card title="Anchor outcomes">
              <dl className="space-y-1.5 text-sm">
                {Object.entries(payload.data.anchor_errors).map(([k, v]) => (
                  <div key={k} className="flex items-baseline justify-between gap-2">
                    <dt className="min-w-0 break-all text-slate-500 dark:text-slate-400">{k}</dt>
                    <dd className="shrink-0 font-semibold tabular-nums">{v}</dd>
                  </div>
                ))}
              </dl>
            </Card>
          </div>

          <Card
            title="Error taxonomy counts"
            subtitle={`${payload.data.n_cases} test cases · multi-label tags`}
            className="mt-4"
          >
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tagData} margin={{ top: 8, right: 8, left: 0, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.3} />
                  <XAxis
                    dataKey="tag"
                    angle={-35}
                    textAnchor="end"
                    interval={0}
                    tick={{ fontSize: 10 }}
                    stroke="#94a3b8"
                  />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} stroke="#94a3b8" />
                  <Tooltip
                    contentStyle={{
                      borderRadius: 8,
                      fontSize: 12,
                      border: '1px solid #cbd5e1',
                    }}
                    formatter={(value) => [String(value), 'count']}
                  />
                  <Bar dataKey="count">
                    {tagData.map((entry) => (
                      <Cell
                        key={entry.tag}
                        fill={
                          entry.tag.startsWith('harmful') || entry.tag.endsWith('_fn') || entry.tag.endsWith('_fp')
                            ? '#dc2626'
                            : entry.tag.startsWith('anchor_followed')
                              ? '#d97706'
                              : entry.tag.startsWith('unchanged_correct') || entry.tag.startsWith('model_tp')
                                ? '#059669'
                                : '#2563eb'
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card
            title="Per-case taxonomy"
            subtitle={payload.rows ? `${payload.rows.length} rows from error_taxonomy.csv` : undefined}
            className="mt-4"
          >
            {payload.rows && payload.rows.length > 0 ? (
              <div className="-mx-4 max-h-[480px] overflow-auto px-4 sm:mx-0 sm:px-0">
                <table className="table-base min-w-[900px]">
                  <thead className="sticky top-0 bg-white dark:bg-slate-900">
                    <tr>
                      <th>row</th>
                      <th>truth</th>
                      <th>model</th>
                      <th>system (control)</th>
                      <th>system (incorrect anchor)</th>
                      <th>model outcome</th>
                      <th>system outcome</th>
                      <th>uncertainty</th>
                      <th>tags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.rows.map((r) => (
                      <tr key={r.row_index}>
                        <td className="tabular-nums">{r.row_index}</td>
                        <td className="text-xs">{r.ground_truth}</td>
                        <td className="text-xs">{r.model_pred}</td>
                        <td className="text-xs">{r.system_pred_control}</td>
                        <td className="text-xs">{r.system_pred_incorrect_anchor}</td>
                        <td>
                          <Badge tone={r.model_outcome === 'TP' || r.model_outcome === 'TN' ? 'green' : 'red'}>
                            {r.model_outcome}
                          </Badge>
                        </td>
                        <td>
                          <Badge tone={r.system_outcome === 'TP' || r.system_outcome === 'TN' ? 'green' : 'red'}>
                            {r.system_outcome}
                          </Badge>
                        </td>
                        <td className="text-xs">{r.uncertainty_level_control}</td>
                        <td className="max-w-[260px] truncate text-xs text-slate-500" title={r.tags}>
                          {r.tags}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState title="No per-case rows" hint="outputs/metrics/error_taxonomy.csv missing" />
            )}
          </Card>
        </>
      )}
    </div>
  )
}

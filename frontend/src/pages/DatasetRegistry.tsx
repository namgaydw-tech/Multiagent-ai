import { Database, ShieldAlert } from 'lucide-react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { Badge, Card, EmptyState, ErrorBox, Loading, PageHeader } from '../components/states'

type BlockingReport = {
  summary?: Record<string, number>
  datasets?: Record<
    string,
    {
      status: string
      reasons?: string[]
      checks?: { key: string; ok: boolean; detail?: string }[]
    }
  >
  n_checks?: number
}

const STATUS_TONE: Record<string, 'green' | 'amber' | 'red' | 'slate'> = {
  PASS: 'green',
  CONDITIONAL: 'amber',
  BLOCKED: 'red',
  NOT_AVAILABLE: 'slate',
}

/** Page 10 — Phase 5 dataset registry: 27-check blocking status + exact reasons. */
export function DatasetRegistry() {
  const status = useAsync(() => api.phase5Status(), [])
  const report = useAsync(() => api.phase5Datasets(), [])

  const data = (report.data?.available ? report.data.data : null) as BlockingReport | null
  const entries = Object.entries(data?.datasets ?? {})

  return (
    <div>
      <PageHeader
        title="Dataset Registry"
        description="Phase 5 candidate datasets with the 27-check pre-training blocker. BLOCKED and NOT_AVAILABLE datasets cannot enter training; mirrors are one dataset, never two cohorts."
      />

      {status.loading && <Loading label="Loading Phase 5 status…" />}
      {status.error && <ErrorBox error={status.error} onRetry={status.reload} />}
      {status.data && (
        <Card
          title="Artifact status"
          subtitle={`Reproduce: ${status.data.reproduce ?? 'python scripts/run_phase5.py analyze'}`}
        >
          <div className="mb-3 flex flex-wrap gap-2">
            {Object.entries(status.data.artifacts).map(([k, ok]) => (
              <Badge key={k} tone={ok ? 'green' : 'slate'}>
                {k}: {ok ? 'GENERATED' : 'missing'}
              </Badge>
            ))}
          </div>
          {status.data.blocker_summary && (
            <div className="flex flex-wrap gap-2">
              {Object.entries(status.data.blocker_summary).map(([k, v]) => (
                <Badge key={k} tone={STATUS_TONE[k] ?? 'slate'}>
                  {k}: {v}
                </Badge>
              ))}
            </div>
          )}
          <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
            Datasets executed end-to-end: {status.data.datasets_executed.join(', ') || 'none'}
          </p>
        </Card>
      )}

      <Card
        title="Blocking report"
        subtitle="Every CONDITIONAL / BLOCKED decision states its exact reasons"
      >
        {report.loading && <Loading label="Loading blocking report…" />}
        {report.error && <ErrorBox error={report.error} onRetry={report.reload} />}
        {!report.data?.available && !report.loading && !report.error && (
          <EmptyState
            title="dataset_blocking_report.json not generated"
            reason={report.data?.reason}
            hint={report.data?.hint ?? 'python scripts/run_phase5.py audit'}
          />
        )}
        {entries.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500 dark:border-slate-700">
                  <th className="py-2 pr-3">Dataset</th>
                  <th className="py-2 pr-3">Status</th>
                  <th className="py-2 pr-3">Reasons / conditions</th>
                </tr>
              </thead>
              <tbody>
                {entries.map(([id, d]) => (
                  <tr
                    key={id}
                    className="border-b border-slate-100 align-top dark:border-slate-800"
                  >
                    <td className="py-2 pr-3 font-mono text-xs">{id}</td>
                    <td className="py-2 pr-3">
                      <Badge tone={STATUS_TONE[d.status] ?? 'slate'}>{d.status}</Badge>
                    </td>
                    <td className="py-2 pr-3 text-xs text-slate-600 dark:text-slate-300">
                      {(d.reasons ?? []).length > 0 ? (
                        <ul className="list-inside list-disc space-y-0.5">
                          {d.reasons!.map((r) => (
                            <li key={r}>{r}</li>
                          ))}
                        </ul>
                      ) : (
                        <span className="text-slate-400">
                          {d.status === 'PASS' ? 'all 27 checks passed' : 'see audit doc'}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-3 flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
          <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
          {data?.n_checks ?? 27}-check blocker · full narrative audit: docs/PHASE5_DATASET_AUDIT.md
        </p>
      </Card>

      <Card title="Modality note" subtitle="What this registry deliberately does not claim">
        <ul className="list-inside list-disc space-y-1 text-sm text-slate-600 dark:text-slate-300">
          <li>
            <Database className="mr-1 inline h-3.5 w-3.5" aria-hidden />
            Regensburg ultrasound shares patients with the tabular cohort — NOT an external
            population.
          </li>
          <li>Kermany on Mendeley and its Kaggle mirror are ONE dataset.</li>
          <li>Unrelated diseases are never concatenated to inflate sample size.</li>
          <li>Credentialed datasets stay NOT_AVAILABLE until real credentials exist.</li>
        </ul>
      </Card>
    </div>
  )
}

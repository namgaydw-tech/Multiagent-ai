import { useState } from 'react'
import { FileJson, ListChecks } from 'lucide-react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { num, pretty } from '../lib/format'
import { Badge, Card, EmptyState, ErrorBox, Loading, PageHeader } from '../components/states'
import type { AuditDetail, AuditListItem } from '../lib/types'

/** Page 8 — browse persisted audit trails: case list → audit detail → stage files. */
export function AuditViewer() {
  const list = useAsync(() => api.audits(200), [])
  const [selected, setSelected] = useState<string | null>(null)

  return (
    <div>
      <PageHeader
        title="Audit Viewer"
        description="Every run persists a full audit trail under outputs/audits/ — stage inputs, outputs, provider metadata, latency and errors. Nothing here is regenerated for display."
      />

      {list.loading && <Loading label="Loading audits…" />}
      {list.error && <ErrorBox error={list.error} onRetry={list.reload} />}

      {list.data && list.data.n === 0 && (
        <EmptyState
          title="No case audits yet"
          reason="Run a case from the Agent Timeline page, or run the Phase 3 script."
          hint="python scripts/run_phase3.py"
        />
      )}

      {list.data && list.data.n > 0 && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)]">
          <Card title={`Persisted cases (${list.data.n})`} subtitle="Newest first">
            <ul className="max-h-[560px] space-y-1.5 overflow-y-auto">
              {list.data.cases.map((c) => (
                <AuditListItemRow
                  key={c.case_id}
                  item={c}
                  active={selected === c.case_id}
                  onSelect={() => setSelected(c.case_id)}
                />
              ))}
            </ul>
          </Card>

          <div>
            {selected === null ? (
              <EmptyState
                title="Select a case audit"
                reason="Shows completed stages, provider metadata, totals, errors and every persisted JSON file."
              />
            ) : (
              <AuditDetailPanel caseId={selected} />
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function AuditListItemRow({
  item,
  active,
  onSelect,
}: {
  item: AuditListItem
  active: boolean
  onSelect: () => void
}) {
  return (
    <li>
      <button
        onClick={onSelect}
        className={`w-full rounded-lg border px-3 py-2 text-left transition ${
          active
            ? 'border-brand-600 bg-brand-50 dark:bg-brand-600/15'
            : 'border-slate-200 hover:border-slate-300 dark:border-slate-700'
        }`}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-mono text-xs font-semibold">{item.case_id}</span>
          {item.has_errors && <Badge tone="red">errors</Badge>}
        </div>
        <div className="mt-1 flex flex-wrap gap-1.5 text-[11px] text-slate-500 dark:text-slate-400">
          {item.ablation && <Badge tone="slate">{item.ablation}</Badge>}
          {item.rag_enabled !== undefined && (
            <Badge tone={item.rag_enabled ? 'green' : 'slate'}>RAG {item.rag_enabled ? 'on' : 'off'}</Badge>
          )}
          {item.completed_stages && <Badge>{item.completed_stages.length} stages</Badge>}
          {item.provider_backend && <Badge tone="blue">{item.provider_backend}</Badge>}
        </div>
      </button>
    </li>
  )
}

function AuditDetailPanel({ caseId }: { caseId: string }) {
  const detail = useAsync(() => api.audit(caseId), [caseId])
  const [file, setFile] = useState<string | null>(null)
  const fileData = useAsync(
    () => (file ? api.auditFile(caseId, file) : Promise.resolve(null)),
    [caseId, file],
  )

  const audit: AuditDetail['audit'] | null = detail.data?.audit ?? null

  return (
    <div className="space-y-4">
      {detail.loading && <Loading label={`Loading audit for ${caseId}…`} />}
      {detail.error && <ErrorBox error={detail.error} onRetry={detail.reload} />}

      {audit && (
        <>
          <Card
            title={caseId}
            subtitle={`started ${audit.started_utc ?? 'n/a'} · completed ${audit.completed_utc ?? 'n/a'}`}
            actions={
              <div className="flex flex-wrap gap-1.5">
                <Badge tone="blue">{audit.provider_backend ?? 'n/a'}</Badge>
                <Badge>{audit.ablation ?? 'default'}</Badge>
                <Badge tone={audit.rag_enabled ? 'green' : 'slate'}>
                  RAG {audit.rag_enabled ? 'on' : 'off'}
                </Badge>
                <Badge tone="green">seed {audit.seed ?? 'n/a'}</Badge>
              </div>
            }
          >
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <p className="text-xs text-slate-500">Latency</p>
                <p className="text-lg font-bold tabular-nums">{num(audit.totals?.latency_s ?? null, 2)}s</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Tokens</p>
                <p className="text-lg font-bold tabular-nums">{audit.totals?.tokens ?? 'n/a'}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Provider calls</p>
                <p className="text-lg font-bold tabular-nums">{audit.totals?.provider_calls ?? 'n/a'}</p>
              </div>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              completed stages: {(audit.completed_stages ?? []).join(', ') || 'none'}
            </p>
            {audit.errors && Object.keys(audit.errors).length > 0 && (
              <div className="mt-2 rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
                {Object.entries(audit.errors).map(([k, v]) => (
                  <p key={k}>{k}: {v}</p>
                ))}
              </div>
            )}
          </Card>

          <Card title="Stage records" subtitle="From audit.json — status, backend, latency, schema">
            <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
              <table className="table-base min-w-[640px]">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Stage</th>
                    <th>Status</th>
                    <th>Backend</th>
                    <th className="text-right">Latency</th>
                    <th>Schema</th>
                    <th>Errors</th>
                  </tr>
                </thead>
                <tbody>
                  {(audit.stages ?? []).map((s) => (
                    <tr key={s.stage}>
                      <td className="tabular-nums">{s.stage}</td>
                      <td>{s.name}</td>
                      <td>
                        <Badge tone={s.status === 'ok' || s.status === 'completed' ? 'green' : 'red'}>
                          {s.status}
                        </Badge>
                      </td>
                      <td className="text-xs">{s.backend ?? '—'}</td>
                      <td className="text-right tabular-nums">
                        {typeof s.latency_s === 'number' ? `${num(s.latency_s, 3)}s` : '—'}
                      </td>
                      <td className="text-xs text-slate-500">{s.schema_name ?? '—'}</td>
                      <td className="max-w-[220px] truncate text-xs text-red-500" title={s.error ?? ''}>
                        {s.error ?? ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card
            title="Persisted files"
            subtitle="Click a file to view its exact persisted JSON"
            actions={
              <button className="btn-secondary !px-2 !py-1 text-xs" onClick={() => setFile(null)} disabled={!file}>
                Close viewer
              </button>
            }
          >
            <div className="flex flex-wrap gap-1.5">
              {(detail.data?.files ?? []).map((f) => (
                <button
                  key={f}
                  onClick={() => setFile(f)}
                  className={`inline-flex items-center gap-1 rounded border px-2 py-1 text-xs font-mono transition ${
                    file === f
                      ? 'border-brand-600 bg-brand-600 text-white'
                      : 'border-slate-300 text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800'
                  }`}
                >
                  <FileJson className="h-3 w-3" aria-hidden />
                  {f}
                </button>
              ))}
            </div>

            {file && (
              <div className="mt-3">
                {fileData.loading && <Loading label={`Loading ${file}…`} />}
                {fileData.error && <ErrorBox error={fileData.error} onRetry={fileData.reload} />}
                {fileData.data && (
                  <pre className="max-h-[420px] overflow-auto rounded bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-100">
                    {pretty(fileData.data.content)}
                  </pre>
                )}
              </div>
            )}
          </Card>

          {audit.disclaimer && (
            <p className="flex items-start gap-1.5 text-xs text-slate-500 dark:text-slate-400">
              <ListChecks className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
              {audit.disclaimer}
            </p>
          )}
        </>
      )}
    </div>
  )
}

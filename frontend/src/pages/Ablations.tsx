import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { pct } from '../lib/format'
import { Badge, Card, EmptyState, ErrorBox, Loading, PageHeader } from '../components/states'
import type { ConditionStats, Headline, MetricRecord } from '../lib/types'

const PROFILE_LABELS: Record<string, string> = {
  '1_lightgbm_only': 'LightGBM only (no LLM)',
  '2_single_llm': 'Single LLM, record only',
  '3_lightgbm_plus_single_llm': 'Single LLM + model',
  '4_lightgbm_self_reflection': 'Model + self-reflection',
  '5_generic_multi_agent': 'Generic multi-agent (no opponent)',
  '6_proponent_opponent': 'Proponent vs opponent',
  '7_full_five_agent': 'Full five agents (no RAG)',
  '8_full_without_rag': 'Full pipeline, RAG off',
  '9_full_with_rag': 'Full pipeline (primary)',
  '10_all_see_anchor': 'All agents see the anchor',
  '11_information_isolated': 'Strict information isolation',
}

function rate(h: Headline | undefined, key: keyof Headline): string {
  const rec = h?.[key] as MetricRecord | undefined
  if (!rec || rec.value === null || rec.value === undefined) return 'n/a'
  return pct(rec.value, 1)
}

function acc(h: Headline | undefined): string {
  return rate(h, 'final_accuracy')
}

/** Page 6 — 11-profile ablation matrix under control and incorrect-anchor settings. */
export function Ablations() {
  const bias = useAsync(() => api.bias(), [])
  const stats: ConditionStats | null = bias.data?.available
    ? (bias.data.data as unknown as ConditionStats)
    : null
  const ablations = stats?.ablations ?? {}
  const profiles = Object.keys(ablations).sort()

  return (
    <div>
      <PageHeader
        title="Ablations"
        description="Which component creates the debiasing effect? Eleven profiles run under a control anchor and an incorrect anchor — compare accuracy, CBR and harmful flips."
      />

      {bias.loading && <Loading label="Loading ablation matrix…" />}
      {bias.error && <ErrorBox error={bias.error} onRetry={bias.reload} />}
      {bias.data && !bias.data.available && (
        <EmptyState
          title="Ablation experiment not generated"
          reason={bias.data.reason}
          hint={bias.data.hint ?? 'python scripts/run_phase4.py ablations --anchor control'}
        />
      )}

      {stats && profiles.length === 0 && (
        <EmptyState
          title="No ablation runs recorded"
          hint="python scripts/run_phase4.py ablations --anchor control && python scripts/run_phase4.py ablations --anchor incorrect_anchor && python scripts/run_phase4.py analyze"
        />
      )}

      {stats && profiles.length > 0 && (
        <>
          <Card
            title="Ablation matrix"
            subtitle={
              stats.ablation_anchor_setting
                ? `anchor setting: ${stats.ablation_anchor_setting}`
                : 'control vs incorrect_anchor'
            }
          >
            <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
              <table className="table-base min-w-[880px]">
                <thead>
                  <tr>
                    <th>Profile</th>
                    <th colSpan={3} className="text-center">Control anchor</th>
                    <th colSpan={3} className="text-center">Incorrect anchor</th>
                    <th className="text-right">Δ accuracy</th>
                  </tr>
                  <tr>
                    <th />
                    <th className="text-right">Acc</th>
                    <th className="text-right">CBR</th>
                    <th className="text-right">HFR</th>
                    <th className="text-right">Acc</th>
                    <th className="text-right">CBR</th>
                    <th className="text-right">HFR</th>
                    <th className="text-right" />
                  </tr>
                </thead>
                <tbody>
                  {profiles.map((p) => {
                    const row = ablations[p]
                    const control = row?.control
                    const wrong = row?.incorrect_anchor
                    const aC = control?.final_accuracy?.value
                    const aW = wrong?.final_accuracy?.value
                    const delta =
                      typeof aC === 'number' && typeof aW === 'number' ? aW - aC : null
                    const primary = p === '9_full_with_rag'
                    return (
                      <tr
                        key={p}
                        className={primary ? 'bg-brand-50/60 dark:bg-brand-600/10' : ''}
                      >
                        <td>
                          <span className="font-medium">{p}</span>
                          {primary && (
                            <Badge tone="blue">
                              <span className="ml-1">primary</span>
                            </Badge>
                          )}
                          <span className="block text-xs text-slate-400">
                            {PROFILE_LABELS[p] ?? ''}
                          </span>
                        </td>
                        <td className="text-right tabular-nums">{acc(control)}</td>
                        <td className="text-right tabular-nums">{rate(control, 'CBR')}</td>
                        <td className="text-right tabular-nums">{rate(control, 'HFR')}</td>
                        <td className="text-right tabular-nums">{acc(wrong)}</td>
                        <td className="text-right tabular-nums">{rate(wrong, 'CBR')}</td>
                        <td className="text-right tabular-nums">{rate(wrong, 'HFR')}</td>
                        <td
                          className={`text-right tabular-nums ${
                            delta !== null && delta < -0.01
                              ? 'text-red-600 dark:text-red-400'
                              : delta !== null && delta > 0.01
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : ''
                          }`}
                        >
                          {delta === null ? 'n/a' : `${delta > 0 ? '+' : ''}${pct(delta, 1)}`}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
              Acc = final accuracy, CBR = confirmation-bias rate (lower is better), HFR =
              harmful-flip rate. Under the deterministic backend, Isolation A ≡ C by
              construction — this is disclosed in docs/PHASE4_REPORT.md, not hidden.
            </p>
          </Card>

          <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card title="Ablation matrix figure">
              <AblationFigure />
            </Card>
            <Card title="How to read this table" subtitle="Mechanism, not just numbers">
              <ul className="list-inside list-disc space-y-1.5 text-sm text-slate-600 dark:text-slate-300">
                <li>
                  <strong>Profile 5 (no opponent)</strong> shows zero harmful flips — the
                  opponent agent is the mechanism that can both correct and flip a case.
                </li>
                <li>
                  <strong>Profiles 2/3 (single agent)</strong> follow incorrect anchors at
                  high rates — anchoring without adversarial challenge.
                </li>
                <li>
                  <strong>Profile 9 (primary)</strong> keeps CBR low while preserving
                  accuracy under anchor pressure.
                </li>
                <li>
                  <strong>RAG on vs off</strong> (8 vs 9) is a descriptive comparison on
                  this dataset, not a powered statistical claim.
                </li>
              </ul>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}

function AblationFigure() {
  const health = useAsync(() => api.health(), [])
  const ok = health.data?.figures.includes('fig_ablation_matrix.png')
  if (health.loading) return <Loading label="Checking figure…" />
  return ok ? (
    <img
      src={api.figureUrl('fig_ablation_matrix.png')}
      alt="Ablation matrix across 11 profiles"
      loading="lazy"
      className="w-full rounded bg-white dark:bg-slate-950"
    />
  ) : (
    <EmptyState
      title="Figure not generated"
      hint="python scripts/run_phase4.py analyze"
    />
  )
}

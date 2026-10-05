import { useMemo, useState } from 'react'
import { FlaskConical, ImageOff, Sigma } from 'lucide-react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import {
  Badge,
  Card,
  EmptyState,
  ErrorBox,
  Loading,
  MetricTile,
  PageHeader,
} from '../components/states'

/** Scorecard label -> metrics_test/metrics_val raw key. */
const METRIC_KEYS: { label: string; key: string }[] = [
  { label: 'Macro F0.5', key: 'macro_f0_5' },
  { label: 'Positive F0.5', key: 'positive_class_f0_5' },
  { label: 'Accuracy', key: 'accuracy' },
  { label: 'Balanced acc.', key: 'balanced_accuracy' },
  { label: 'Precision', key: 'macro_precision' },
  { label: 'Sensitivity', key: 'sensitivity' },
  { label: 'Specificity', key: 'specificity' },
  { label: 'NPV', key: 'npv' },
  { label: 'F1', key: 'macro_f1' },
  { label: 'F2', key: 'macro_f2' },
  { label: 'MCC', key: 'mcc' },
  { label: 'AUROC', key: 'auroc' },
  { label: 'AUPRC', key: 'auprc' },
  { label: 'Brier', key: 'brier' },
  { label: 'ECE', key: 'ece' },
]

const FIGURE_SELECT: { id: number; label: string }[] = [
  { id: 9, label: 'ROC curves' },
  { id: 10, label: 'Precision-recall' },
  { id: 11, label: 'Confusion matrix' },
  { id: 12, label: 'Normalized confusion' },
  { id: 13, label: 'Calibration / reliability' },
  { id: 14, label: 'Threshold vs Macro F0.5' },
  { id: 15, label: 'Threshold vs sensitivity' },
  { id: 16, label: 'Threshold vs specificity' },
  { id: 17, label: 'Threshold vs precision' },
  { id: 18, label: 'Operating point' },
  { id: 19, label: 'Learning curves' },
  { id: 20, label: 'Feature importance (native)' },
  { id: 21, label: 'Permutation importance' },
  { id: 22, label: 'SHAP summary' },
  { id: 23, label: 'SHAP waterfall' },
  { id: 27, label: 'RF n_estimators curve' },
  { id: 28, label: 'KNN K curve' },
  { id: 1, label: 'Macro F0.5 leaderboard' },
  { id: 30, label: 'Ensemble vs individual' },
  { id: 35, label: 'Confidence intervals' },
  { id: -1, label: 'Grad-CAM / image-model figures' },
]

type DatasetMetrics = Awaited<ReturnType<typeof api.phase5DatasetMetrics>> | null

function fmt(v: unknown): string {
  if (v === null || v === undefined) return 'n/a'
  return typeof v === 'number' ? (Number.isInteger(v) && Math.abs(v) >= 100 ? String(v) : v.toFixed(3)) : String(v)
}

/** Hyperparameters arrive as an object of best_params — render them readably. */
function fmtHyper(v: unknown): string {
  if (v === null || v === undefined) return 'n/a'
  if (typeof v === 'string') return v
  if (typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>)
    if (entries.length === 0) return '{} (defaults)'
    return entries.map(([k, val]) => `${k}=${JSON.stringify(val)}`).join(', ')
  }
  return String(v)
}

/** Page 12 — Algorithm Lab: dataset × algorithm × metric with formulas + figures. */
export function AlgorithmLab() {
  const status = useAsync(() => api.phase5Status(), [])
  const catalogue = useAsync(() => api.phase5Algorithms(), [])
  const manifest = useAsync(() => api.phase5Figures(), [])
  const scorecard = useAsync(() => api.phase5Scorecard(), [])

  const [dataset, setDataset] = useState('')
  const [algorithm, setAlgorithm] = useState('gradient_boosting')
  const [metric, setMetric] = useState('macro_f0_5')
  const [figId, setFigId] = useState(9)

  const executed = status.data?.datasets_executed ?? []
  const activeDataset = dataset || executed[0] || ''

  const dm = useAsync<DatasetMetrics>(
    () => (activeDataset ? api.phase5DatasetMetrics(activeDataset) : Promise.resolve(null)),
    [activeDataset],
  )
  const dir = dm.data?.dir ?? ''

  const figEntry = manifest.data?.data?.figures?.find((f) => f.id === figId)

  const figMeta = useAsync(
    () =>
      figEntry?.status === 'GENERATED'
        ? api.phase5FigureMeta(figId, dir || undefined)
        : Promise.resolve(null),
    [figId, dir, figEntry?.status],
  )

  const algorithms = useMemo(() => {
    const test = dm.data?.test
    if (!test) return [] as { name: string; kind: 'model' | 'ensemble' }[]
    const models = Object.entries(test.models ?? {})
      .filter(([, e]) => e.available)
      .map(([name]) => ({ name, kind: 'model' as const }))
    const ensembles = Object.entries(test.ensembles ?? {})
      .filter(([, e]) => e.available)
      .map(([name]) => ({ name, kind: 'ensemble' as const }))
    return [...models, ...ensembles]
  }, [dm.data])

  const active = algorithms.find((a) => a.name === algorithm)
  const testEntry =
    dm.data?.test?.models?.[algorithm] ?? dm.data?.test?.ensembles?.[algorithm]
  const valEntry = dm.data?.validation?.metrics?.[algorithm]
  const threshold =
    dm.data?.thresholds?.thresholds?.[algorithm]?.threshold ?? testEntry?.threshold ?? null
  const calSelected = dm.data?.calibration?.[algorithm]?.selected ?? null

  const scoreRow = scorecard.data?.data?.rows.find(
    (r) => r.dataset === activeDataset && r.algorithm === algorithm,
  )

  const formula = catalogue.data?.algorithms?.[algorithm]
  const selectedMetric = METRIC_KEYS.find((m) => m.key === metric)

  return (
    <div>
      <PageHeader
        title="Algorithm Lab"
        description="Pick a dataset, an algorithm and a metric: formula, hyperparameters, locked threshold, validation vs sealed-test metrics, and the persisted figures. Missing artifacts show NOT GENERATED with the reproduce command — nothing is fabricated."
      />

      {status.error && <ErrorBox error={status.error} onRetry={status.reload} />}

      <Card title="Selections" subtitle={catalogue.data?.selection_rule ?? ''}>
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="block text-xs font-semibold text-slate-500">
            Dataset
            <select
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-600 dark:bg-slate-800"
              value={activeDataset}
              onChange={(e) => setDataset(e.target.value)}
            >
              {executed.length === 0 && <option value="">(loading…)</option>}
              {executed.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs font-semibold text-slate-500">
            Algorithm
            <select
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-600 dark:bg-slate-800"
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
            >
              {algorithms.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                  {a.kind === 'ensemble' ? ' (ensemble)' : ''}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs font-semibold text-slate-500">
            Metric
            <select
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-600 dark:bg-slate-800"
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
            >
              {METRIC_KEYS.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </Card>

      {dm.loading && <Loading label="Loading dataset metrics…" />}
      {dm.error && <ErrorBox error={dm.error} onRetry={dm.reload} />}
      {!dm.data && !dm.loading && !dm.error && (
        <EmptyState
          title="Select a dataset"
          hint="python scripts/run_phase5.py train --dataset <ID> && python scripts/run_phase5.py evaluate --dataset <ID>"
        />
      )}

      {dm.data && (
        <>
          <Card
            title={`${algorithm} — ${formula?.family ?? active?.kind ?? ''}`}
            subtitle={formula?.why ?? 'No catalogue entry for this algorithm.'}
          >
            <div className="mb-3 flex flex-wrap gap-2">
              <Badge tone="blue">{activeDataset}</Badge>
              <Badge tone={active?.kind === 'ensemble' ? 'amber' : 'green'}>
                {active?.kind ?? 'model'}
              </Badge>
              <Badge>calibration: {calSelected ?? 'n/a'}</Badge>
              <Badge>threshold: {fmt(threshold)}</Badge>
              <Badge tone="green">validation + sealed test evaluated</Badge>
              {scoreRow && <Badge>splits: {dm.data.split_source ?? 'n/a'}</Badge>}
            </div>

            <p className="mb-3 flex items-start gap-2 rounded-lg bg-slate-100 p-3 font-mono text-xs text-slate-700 dark:bg-slate-800 dark:text-slate-200">
              <Sigma className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              {formula?.formula ?? 'formula: see docs/ML_ALGORITHMS_AND_MATHEMATICS.md'}
            </p>

            <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricTile
                label={selectedMetric?.label ?? 'Macro F0.5'}
                value={fmt(testEntry?.test_metrics?.[metric] ?? valEntry?.[metric])}
                tone="good"
              />
              <MetricTile label="Validation" value={fmt(valEntry?.[metric])} />
              <MetricTile label="Sealed test" value={fmt(testEntry?.test_metrics?.[metric])} />
              <MetricTile
                label="Training time (s)"
                value={fmt(scoreRow?.training_time ?? testEntry?.training_seconds)}
              />
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 uppercase tracking-wide text-slate-500 dark:border-slate-700">
                    <th className="py-2 pr-3">Metric</th>
                    <th className="py-2 pr-3">Validation</th>
                    <th className="py-2 pr-3">Sealed test</th>
                    <th className="py-2 pr-3">Test 95% CI</th>
                  </tr>
                </thead>
                <tbody>
                  {METRIC_KEYS.map((m) => {
                    const ci = testEntry?.bootstrap_ci?.[m.key]
                    return (
                      <tr
                        key={m.key}
                        className={`border-b border-slate-100 dark:border-slate-800 ${
                          m.key === metric ? 'bg-brand-50 dark:bg-brand-900/20' : ''
                        }`}
                      >
                        <td className="py-1.5 pr-3 font-medium">{m.label}</td>
                        <td className="py-1.5 pr-3 tabular-nums">
                          {fmt(valEntry?.[m.key])}
                        </td>
                        <td className="py-1.5 pr-3 tabular-nums">
                          {fmt(testEntry?.test_metrics?.[m.key])}
                        </td>
                        <td className="py-1.5 pr-3 tabular-nums text-slate-500">
                          {ci ? `[${fmt(ci.lo)}, ${fmt(ci.hi)}]` : 'n/a'}
                        </td>
                      </tr>
                    )
                  })}
                  <tr className="border-b border-slate-100 dark:border-slate-800">
                    <td className="py-1.5 pr-3 font-medium">Threshold</td>
                    <td className="py-1.5 pr-3 tabular-nums">{fmt(threshold)}</td>
                    <td className="py-1.5 pr-3 tabular-nums">{fmt(threshold)}</td>
                    <td className="py-1.5 pr-3 text-slate-500">locked on validation</td>
                  </tr>
                  <tr>
                    <td className="py-1.5 pr-3 font-medium">Inference (ms/case)</td>
                    <td className="py-1.5 pr-3">n/a</td>
                    <td className="py-1.5 pr-3 tabular-nums">
                      {fmt(testEntry?.inference_ms_per_case)}
                    </td>
                    <td className="py-1.5 pr-3 text-slate-500">per-case, test partition</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
              Hyperparameters: {fmtHyper(scoreRow?.hyperparameters)} · floor pass:{' '}
              {scoreRow?.sensitivity_floor_pass === null || scoreRow?.sensitivity_floor_pass === undefined
                ? 'n/a'
                : String(scoreRow.sensitivity_floor_pass)}{' '}
              · n(test) = {fmt(dm.data.counts?.test?.n ?? null)}
            </p>
          </Card>

          <Card
            title="Figures"
            subtitle="Every image below comes from outputs/phase5 — never invented"
          >
            <label className="mb-3 block text-xs font-semibold text-slate-500">
              Graph selection
              <select
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm dark:border-slate-600 dark:bg-slate-800"
                value={figId}
                onChange={(e) => setFigId(Number(e.target.value))}
              >
                {FIGURE_SELECT.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.label}                      {figEntryFor(manifest.data?.data?.figures, f.id)?.status === 'GENERATED'
                      ? ''
                      : ' — NOT GENERATED'}
                  </option>
                ))}
              </select>
            </label>

            {manifest.loading && <Loading label="Loading figure manifest…" />}
            {manifest.error && <ErrorBox error={manifest.error} onRetry={manifest.reload} />}
            {figId === -1 && (
              <EmptyState
                title="NOT GENERATED — Grad-CAM / image-model figures"
                reason="Image pipeline NOT_EXECUTED: torch is not installed in this environment (Kermany CXR / Regensburg ultrasound models were not trained). No image results are invented."
                hint="pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && python scripts/run_phase5.py all"
              />
            )}
            {figId !== -1 && figEntry?.status === 'GENERATED' && (
              <>
                {figMeta.loading && <Loading label="Loading figure…" />}
                {figMeta.error && <ErrorBox error={figMeta.error} onRetry={figMeta.reload} />}
                {figMeta.data?.url && (
                  <figure>
                    <img
                      src={figMeta.data.url}
                      alt={figEntry.name}
                      className="max-w-full rounded-lg border border-slate-200 dark:border-slate-700"
                      loading="lazy"
                    />
                    <figcaption className="mt-1 text-xs text-slate-500">
                      {figEntry.name} — persisted PNG (research figure, seed 20261002)
                    </figcaption>
                  </figure>
                )}
              </>
            )}
            {figId !== -1 && figEntry?.status !== 'GENERATED' && (
              <EmptyState
                title={`NOT GENERATED — ${figEntry?.name ?? 'figure'}`}
                reason={figEntry?.note}
                hint={figEntry?.reproduce ?? manifest.data?.data?.reproduce}
              />
            )}
            <div className="mt-3 flex items-start gap-2 text-xs text-slate-500 dark:text-slate-400">
              <ImageOff className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
              Image-model figures (Grad-CAM, CXR ROC) are NOT_EXECUTED — torch is not
              installed in this environment; no image results are invented.
            </div>
          </Card>

          <Card
            title="Metric formulas"
            subtitle="Static mathematics documentation — never performance numbers"
          >
            <div className="grid gap-2 sm:grid-cols-2">
              {Object.entries(catalogue.data?.metrics ?? {}).map(([k, f]) => (
                <p
                  key={k}
                  className="rounded bg-slate-100 px-2 py-1 font-mono text-xs dark:bg-slate-800"
                >
                  <span className="font-semibold">{k}</span> = {f}
                </p>
              ))}
            </div>
            <p className="mt-2 flex items-center gap-1.5 text-xs text-slate-500">
              <FlaskConical className="h-3.5 w-3.5" aria-hidden />
              Research prototype — not a medical device.
            </p>
          </Card>
        </>
      )}
    </div>
  )
}

function figEntryFor(
  figures: { id: number; status: string }[] | undefined,
  id: number,
): { status: string } | undefined {
  return figures?.find((f) => f.id === id)
}

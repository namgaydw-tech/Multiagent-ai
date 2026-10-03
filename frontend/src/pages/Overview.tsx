import { Link } from 'react-router-dom'
import { ArrowRight, FlaskConical, Gauge, ShieldCheck } from 'lucide-react'
import { api } from '../lib/api'
import { useAsync } from '../hooks/useAsync'
import { num, pct } from '../lib/format'
import { Badge, Card, ErrorBox, Loading, MetricTile, PageHeader } from '../components/states'

/** Page 1 — service health, headline results, navigation. */
export function Overview() {
  const health = useAsync(() => api.health(), [])
  const finalResults = useAsync(() => api.finalResults(), [])

  const fr = finalResults.data
  const bias = fr?.available ? (fr.data as Record<string, unknown>)?.['bias_reduction'] : null
  const br = bias as Record<string, unknown> | null
  const cbrMulti = (br?.['cbr_multi'] ?? null) as Record<string, unknown> | null
  const diff = (br?.['paired_diff'] ?? null) as Record<string, unknown> | null
  const phase2 = (fr?.available
    ? (fr.data as Record<string, unknown>)?.['phase2_reference_test_metrics']
    : null) as Record<string, number> | null

  return (
    <div>
      <PageHeader
        title="Overview"
        description="Mitigating confirmation bias in automated pediatric diagnostics via adversarial multi-agent AI — live results from this repository."
      />

      {health.loading && <Loading label="Contacting backend…" />}
      {health.error && <ErrorBox error={health.error} onRetry={health.reload} />}

      {health.data && (
        <Card title="System status" subtitle={health.data.disclaimer}>
          <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
            <div>
              <p className="text-slate-500 dark:text-slate-400">Model</p>
              <p className="font-semibold">
                {health.data.model.available ? health.data.model.name : 'not available'}
              </p>
              <p className="text-xs text-slate-500">
                v{health.data.model.version} · {health.data.model.calibrator ?? '—'} ·
                threshold {health.data.model.threshold ?? 'n/a'}
              </p>
            </div>
            <div>
              <p className="text-slate-500 dark:text-slate-400">LLM backend</p>
              <p className="font-semibold">{health.data.llm_backend}</p>
              <p className="text-xs text-slate-500">
                {health.data.llm_configured ? 'API key configured' : 'no API key — deterministic'}
              </p>
            </div>
            <div>
              <p className="text-slate-500 dark:text-slate-400">RAG index</p>
              <p className="font-semibold">{health.data.rag_chunks ?? 'n/a'} chunks</p>
              <p className="text-xs text-slate-500">offline TF-IDF + FTS5</p>
            </div>
            <div>
              <p className="text-slate-500 dark:text-slate-400">Artifacts</p>
              <p className="font-semibold">
                {Object.values(health.data.metrics_files).filter(Boolean).length}/
                {Object.keys(health.data.metrics_files).length} metrics files
              </p>
              <p className="text-xs text-slate-500">
                {health.data.figures.length} figures generated
              </p>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge tone="blue">seed {health.data.seed}</Badge>
            <Badge tone="green">config {health.data.model.config_hash ?? 'n/a'}</Badge>
            {Object.entries(health.data.experiments_available).map(([k, ok]) => (
              <Badge key={k} tone={ok ? 'green' : 'amber'}>
                {k}: {ok ? 'executed' : 'missing'}
              </Badge>
            ))}
          </div>
        </Card>
      )}

      <div className="mt-4">
        {finalResults.loading && <Loading label="Loading headline results…" />}
        {finalResults.error && <ErrorBox error={finalResults.error} onRetry={finalResults.reload} />}
        {fr && !fr.available && (
          <Card title="Headline results">
            <p className="text-sm text-slate-500">{fr.reason}</p>
            {fr.hint && <code className="mt-2 block text-xs">{fr.hint}</code>}
          </Card>
        )}
        {fr?.available && (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricTile
              label="Test AUROC (LightGBM)"
              value={num(phase2?.auroc, 3)}
              sub="Phase 2 · 2000-bootstrap CI in Model page"
              tone="good"
            />
            <MetricTile
              label="Test AUPRC (LightGBM)"
              value={num(phase2?.auprc, 3)}
              sub="positive class ≈ 41%"
            />
            <MetricTile
              label="CBR — multi-agent"
              value={pct(typeof cbrMulti?.value === 'number' ? cbrMulti.value : null, 1)}
              sub="incorrect anchors, N=117×3"
              tone="good"
            />
            <MetricTile
              label="Bias reduction vs single"
              value={pct(typeof diff?.diff === 'number' ? diff.diff : null, 1)}
              sub={`paired p=${typeof diff?.p_value === 'number' ? diff.p_value : 'n/a'}`}
              tone="good"
            />
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        <QuickLink to="/cases" title="Run the model on a case"
          text="Real LightGBM prediction + calibration + SHAP for any test row." />
        <QuickLink to="/agents" title="Watch the 8-stage debate"
          text="Information isolation, watchdog, arbitration — with per-stage audit." />
        <QuickLink to="/bias" title="Confirmation-bias experiments"
          text="CBR/AOR/BCR/HFR with bootstrap CIs and significance tests." />
      </div>

      <Card title="Safety framing" className="mt-4">
        <div className="flex items-start gap-2 text-sm text-slate-600 dark:text-slate-300">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" aria-hidden />
          <p>
            This system is a <strong>research prototype</strong> for studying confirmation bias.
            It is <strong>not a medical device</strong>, not a diagnosis, and must not be used
            for clinical decisions. All results derive from the legally permitted Regensburg
            pediatric appendicitis research dataset.
          </p>
        </div>
        <div className="mt-3 flex flex-wrap gap-3 text-sm">
          <Link className="inline-flex items-center gap-1 text-brand-600 hover:underline dark:text-brand-400"
                to="/repro">
            <Gauge className="h-4 w-4" aria-hidden /> Reproduce everything
          </Link>
          <Link className="inline-flex items-center gap-1 text-brand-600 hover:underline dark:text-brand-400"
                to="/ablations">
            <FlaskConical className="h-4 w-4" aria-hidden /> 11 ablation profiles
          </Link>
          <Link className="inline-flex items-center gap-1 text-brand-600 hover:underline dark:text-brand-400"
                to="/errors">
            <ArrowRight className="h-4 w-4" aria-hidden /> Where the system fails
          </Link>
        </div>
      </Card>
    </div>
  )
}

function QuickLink({ to, title, text }: { to: string; title: string; text: string }) {
  return (
    <Link
      to={to}
      className="card group flex items-start justify-between gap-2 transition hover:border-brand-500 hover:shadow"
    >
      <div>
        <p className="font-semibold text-slate-800 dark:text-slate-100">{title}</p>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{text}</p>
      </div>
      <ArrowRight
        className="mt-1 h-4 w-4 shrink-0 text-slate-400 transition group-hover:translate-x-0.5 group-hover:text-brand-500"
        aria-hidden
      />
    </Link>
  )
}

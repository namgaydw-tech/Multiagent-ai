/** Types mirroring the FastAPI responses in `backend/schemas`. */

export interface MetricRecord {
  value: number | null
  n_num?: number
  n_den?: number
  n_excluded?: number
  ci95?: (number | null)[] | null
  [key: string]: unknown
}

export interface DiversityRecord {
  mean_candidates_per_case: number | null
  n_cases?: number
  unique_hypotheses_corpus?: number
  corpus_hypotheses?: string[]
}

export interface Headline {
  n_rows?: number
  CBR: MetricRecord
  AOR: MetricRecord
  BCR: MetricRecord
  HFR: MetricRecord
  CRR: MetricRecord
  CMR: MetricRecord
  DR: MetricRecord
  DDR: DiversityRecord
  final_accuracy: MetricRecord
  initial_accuracy: MetricRecord
}

export interface ModelInfo {
  name: string
  version: string
  available: boolean
  calibrator: string | null
  threshold: number | null
  config_hash: string | null
}

export interface Health {
  status: string
  seed: number
  llm_backend: string
  llm_configured: boolean
  model: ModelInfo
  rag_chunks: number | null
  metrics_files: Record<string, boolean>
  figures: string[]
  experiments_available: Record<string, boolean>
  disclaimer: string
}

export interface Availability<T> {
  available: boolean
  reason?: string
  hint?: string
  data?: T
  disclaimer?: string
}

export interface CaseRow {
  row_index: number
  y_true: number
  p_raw: number
  p_calibrated: number
  threshold: number
  pred_class: string
  uncertainty_level: string
  entropy: number
  model_version: string | null
  config_hash: string | null
}

export interface ShapContributor {
  feature: string
  value: number | string | null
  contribution: number
  direction: 'positive' | 'negative'
  importance: number
}

export interface PredictResponse {
  row_index: number
  model_output: {
    model_name: string
    model_version: string
    class_probabilities: Record<string, number>
    predicted_class: string
    calibrated_probability: number | null
    config_hash: string
    [key: string]: unknown
  }
  shap_explain: {
    case_id: string
    wording_rule: string
    top_contributors: ShapContributor[]
    raw_probability: number
    [key: string]: unknown
  }
  uncertainty: {
    confidence: number
    predictive_entropy: number
    margin: number
    uncertainty_level: 'LOW' | 'MODERATE' | 'HIGH'
    note?: string
  }
}

export interface StageSummary {
  stage: number
  key: string
  name: string
  backend: string | null
  model_version: string | null
  prompt_hash: string | null
  latency_s: number | null
  visibility: { sees_prediction: boolean; sees_anchor: boolean; sees_ground_truth: boolean }
  withheld: string[]
  output: Record<string, unknown>
}

export interface AgentRunResponse {
  case_id: string
  row_index: number
  ground_truth: string | null
  anchor: { value?: string; source?: string; confidence?: number; method?: string } | null
  ablation: string
  rag_enabled: boolean
  summary: {
    model_top?: string
    model_probability?: number
    primary_working_diagnosis?: string
    multi_agent_confidence?: number
    watchdog_risk?: string
    completed_stages?: number[]
    errors?: Record<string, string>
    [key: string]: unknown
  }
  stages: StageSummary[]
  audit_path: string
  elapsed_s: number
  disclaimer: string
}

export interface ConditionStats {
  anchor_experiment: { profile: string; n_cases: number; reps: number; conditions: Record<string, Headline> }
  single_baselines: Record<string, Record<string, Headline>>
  bias_reduction: {
    definition: string
    cbr_single_2: MetricRecord
    cbr_single_3: MetricRecord
    cbr_multi: MetricRecord
    paired_diff: MetricRecord & { p_value?: number; n_pairs?: number }
    mcnemar: { b?: number; c?: number; n_discordant?: number; p_value?: number | null }
  }
  statistics: {
    comparisons: {
      name: string
      test: string
      p_value?: number | null
      q_value?: number | null
      rejected_bh?: boolean
      diff?: number
      [key: string]: unknown
    }[]
    benjamini_hochberg: {
      q_values: (number | null)[]
      rejected: boolean[]
      alpha: number
      n_tests: number
    }
  }
  ablations: Record<string, Record<string, Headline>>
  ablation_anchor_setting: string | null
}

export interface ErrorRow {
  row_index: number
  ground_truth: string
  model_pred: string
  system_pred_control: string
  system_pred_incorrect_anchor: string
  model_outcome: string
  system_outcome: string
  delta_control: string
  anchor_outcome: string | null
  uncertainty_level_control: string
  watchdog_risk_control: string
  tags: string
}

export interface ErrorAnalysis {
  available?: boolean
  reason?: string
  hint?: string
  data?: {
    n_cases: number
    tag_counts: Record<string, number>
    rates: Record<string, number | null>
    fn_breakdown: Record<string, number>
    fp_breakdown: Record<string, number>
    anchor_errors: Record<string, number>
  }
  rows?: ErrorRow[]
}

export interface AuditListItem {
  case_id: string
  audit_available: boolean
  files: string[]
  completed_stages?: number[]
  provider_backend?: string
  ablation?: string
  rag_enabled?: boolean
  started_utc?: string
  has_errors?: boolean
  reason?: string
}

export interface AuditStageEntry {
  stage: number
  name: string
  status: string
  backend?: string | null
  model_version?: string | null
  prompt_hash?: string | null
  input_tokens?: number
  output_tokens?: number
  latency_s?: number
  attempts?: number
  schema_name?: string
  error?: string | null
}

export interface AuditDetail {
  case_id: string
  files: string[]
  audit: {
    case_id: string
    config_hash?: string
    seed?: number
    ablation?: string
    rag_enabled?: boolean
    provider_backend?: string
    completed_stages: number[]
    errors: Record<string, string>
    totals: { latency_s: number; tokens: number; provider_calls: number }
    stages: AuditStageEntry[]
    visibility_policy?: Record<string, unknown>
    started_utc?: string
    completed_utc?: string
    disclaimer?: string
  }
}

export interface ReproCommands {
  train: string[]
  agents_cases: string[]
  experiments: string[]
  tests: string[]
  backend: string[]
  frontend: string[]
  seed: number
  disclaimer: string
}

export interface QuickExperimentResponse {
  n_cases: number
  conditions: string[]
  profiles: string[]
  rows: Record<string, unknown>[]
  headline: Record<string, Headline>
  elapsed_s: number
  note: string
  disclaimer: string
}

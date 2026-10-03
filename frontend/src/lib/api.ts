/** Typed API client with honest error normalization.

Every failure surfaces as an :class:`ApiError` carrying the backend's real
message plus its actionable ``hint`` (e.g. "Run: python scripts/run_phase2.py").
Availability states (``available: false`` with a reason) are returned as normal
payloads so pages can render explicit empty states instead of fake data.
 */

import type {
  AgentRunResponse,
  AuditDetail,
  AuditListItem,
  Availability,
  CaseRow,
  ConditionStats,
  ErrorAnalysis,
  Health,
  PredictResponse,
  QuickExperimentResponse,
  ReproCommands,
} from './types'

export class ApiError extends Error {
  status: number
  hint?: string

  constructor(message: string, status: number, hint?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.hint = hint
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch (err) {
    throw new ApiError(
      'Backend unreachable — is it running on port 8765?',
      0,
      'python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765',
    )
  }
  let body: unknown = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok) {
    const detail: unknown = (body as Record<string, unknown> | null)?.detail
    let message = `HTTP ${res.status}`
    let hint: string | undefined
    if (typeof detail === 'string') {
      message = detail
    } else if (detail && typeof detail === 'object') {
      const d = detail as Record<string, unknown>
      if (typeof d.detail === 'string') message = d.detail
      if (typeof d.hint === 'string') hint = d.hint
    }
    const top = body as Record<string, unknown> | null
    if (!hint && top && typeof top.hint === 'string') hint = top.hint
    throw new ApiError(message, res.status, hint)
  }
  return body as T
}

export interface ExperimentQuickBody {
  conditions: string[]
  profiles: string[]
  n_cases: number
}

export const api = {
  health: () => request<Health>('/api/health'),
  repro: () => request<ReproCommands>('/api/repro'),
  summary: () => request<Availability<Record<string, unknown>>>('/api/metrics/summary'),
  bias: () => request<Availability<ConditionStats>>('/api/metrics/bias'),
  finalResults: () => request<Availability<Record<string, unknown>>>('/api/metrics/final-results'),
  errorAnalysis: () => request<ErrorAnalysis>('/api/metrics/error-analysis'),
  experiments: () => request<Availability<ConditionStats>>('/api/metrics/experiments'),
  experimentStatus: () => request<Record<string, unknown>>('/api/experiments/status'),
  cases: (limit = 100) => request<{ available: boolean; n: number; rows: CaseRow[]; reason?: string; hint?: string }>(`/api/cases?limit=${limit}`),
  caseRecord: (row: number) => request<{ row_index: number; fields: Record<string, string | number | null>; n_fields: number; ground_truth: string }>(`/api/cases/${row}/record`),
  predict: (row_index: number) =>
    request<PredictResponse>('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ row_index }),
    }),
  runAgents: (body: {
    row_index: number
    ablation?: string
    rag_enabled?: boolean
    anchor_condition?: string | null
  }) =>
    request<AgentRunResponse>('/api/agents/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  quickExperiment: (body: ExperimentQuickBody) =>
    request<QuickExperimentResponse>('/api/experiments/quick', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  audits: (limit = 100) =>
    request<{ available: boolean; n: number; cases: AuditListItem[] }>(`/api/audits?limit=${limit}`),
  audit: (caseId: string) =>
    request<AuditDetail>(`/api/audits/${encodeURIComponent(caseId)}`),
  auditFile: (caseId: string, file: string) =>
    request<{ case_id: string; file: string; content: unknown }>(
      `/api/audits/${encodeURIComponent(caseId)}/${encodeURIComponent(file)}`,
    ),
  figureUrl: (name: string) => `/api/metrics/figures/${name}`,
}

export const ANCHOR_CONDITIONS = [
  'control',
  'correct_anchor',
  'incorrect_anchor',
  'high_confidence_incorrect_anchor',
  'low_confidence_incorrect_anchor',
  'model_anchor',
  'no_model_anchor',
] as const

export const PROFILES = [
  '1_lightgbm_only',
  '2_single_llm',
  '3_lightgbm_plus_single_llm',
  '4_lightgbm_self_reflection',
  '5_generic_multi_agent',
  '6_proponent_opponent',
  '7_full_five_agent',
  '8_full_without_rag',
  '9_full_with_rag',
  '10_all_see_anchor',
  '11_information_isolated',
] as const

export const ABLATIONS = [
  'default',
  'A_all_see_prediction',
  'B_only_proponent',
  'C_hidden_until_differential_complete',
  'all_see_anchor',
] as const

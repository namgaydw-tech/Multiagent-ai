import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../lib/api'

interface AsyncState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

/** Small data-fetching hook: loading / error (with hint) / data + manual reload. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)
  const fnRef = useRef(fn)
  fnRef.current = fn

  useEffect(() => {
    let alive = true
    setLoading(true)
    fnRef
      .current()
      .then((value) => {
        if (alive) {
          setData(value)
          setError(null)
        }
      })
      .catch((err: unknown) => {
        if (alive) {
          setError(
            err instanceof ApiError
              ? err
              : new ApiError(err instanceof Error ? err.message : String(err), 0),
          )
        }
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, reload }
}

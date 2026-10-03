import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

function currentTheme(): Theme {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

/** Dark/light theme with localStorage persistence (matches index.html boot script).
 *
 * The ``toggle`` callback computes the next value from the DOM, persists it and
 * sets state with a *plain* value: under React Strict Mode (development) the
 * functional updater form is invoked twice, and a side-effectful updater would
 * toggle the class twice and cancel itself out. Applying the class happens in
 * an effect keyed on ``theme`` instead. The OS preference keeps being followed
 * until the user explicitly toggles (only then is the choice persisted).
 */
export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(currentTheme)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => {
      const stored = localStorage.getItem('theme')
      if (stored === 'light' || stored === 'dark') return
      setTheme(media.matches ? 'dark' : 'light')
    }
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  const toggle = useCallback(() => {
    const next: Theme = currentTheme() === 'dark' ? 'light' : 'dark'
    localStorage.setItem('theme', next)
    setTheme(next)
  }, [])

  return { theme, toggle }
}

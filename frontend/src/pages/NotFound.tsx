import { Link } from 'react-router-dom'
import { Compass } from 'lucide-react'
import { PageHeader } from '../components/states'

/** Unknown route — explicit 404 with navigation back into the app. */
export function NotFound() {
  return (
    <div>
      <PageHeader title="Page not found" />
      <div className="card flex flex-col items-center gap-3 py-10 text-center">
        <Compass className="h-10 w-10 text-slate-400" aria-hidden />
        <p className="text-sm text-slate-500 dark:text-slate-400">
          This route does not exist in the research UI.
        </p>
        <Link className="btn" to="/">
          Back to Overview
        </Link>
      </div>
    </div>
  )
}

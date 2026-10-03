import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Overview } from './pages/Overview'
import { ModelPerformance } from './pages/ModelPerformance'
import { CaseExplorer } from './pages/CaseExplorer'
import { AgentTimeline } from './pages/AgentTimeline'
import { BiasExperiments } from './pages/BiasExperiments'
import { Ablations } from './pages/Ablations'
import { ErrorAnalysisPage } from './pages/ErrorAnalysisPage'
import { AuditViewer } from './pages/AuditViewer'
import { Reproduction } from './pages/Reproduction'
import { NotFound } from './pages/NotFound'

/** Nine-page research UI; every page handles loading, errors and missing data. */
export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="model" element={<ModelPerformance />} />
        <Route path="cases" element={<CaseExplorer />} />
        <Route path="agents" element={<AgentTimeline />} />
        <Route path="bias" element={<BiasExperiments />} />
        <Route path="ablations" element={<Ablations />} />
        <Route path="errors" element={<ErrorAnalysisPage />} />
        <Route path="audits" element={<AuditViewer />} />
        <Route path="repro" element={<Reproduction />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}

import React, { useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import {
  BarChart3, Boxes, BrainCircuit, Database, FlaskConical, Gauge, Menu, Moon, PenLine, Sun, Target, Users,
} from "lucide-react";
import { StudentProvider } from "./api/StudentContext.jsx";
import { ErrorBoundary } from "./components/ui.jsx";
import StudentSelector from "./components/StudentSelector.jsx";
import UploadPage from "./pages/UploadPage.jsx";
import AnalysisPage from "./pages/AnalysisPage.jsx";
import PracticePage from "./pages/PracticePage.jsx";
import ProgressPage from "./pages/ProgressPage.jsx";
import TeacherPage from "./pages/TeacherPage.jsx";
import LabPage from "./pages/LabPage.jsx";
import LabModelPage from "./pages/LabModelPage.jsx";
import PlaygroundPage from "./pages/PlaygroundPage.jsx";
import DatasetPage from "./pages/DatasetPage.jsx";
import BenchmarkPage from "./pages/BenchmarkPage.jsx";

const NAV = [
  { title: "Student", links: [
    { to: "/", label: "New work", icon: PenLine, end: true },
    { to: "/practice", label: "Practice", icon: Target },
    { to: "/progress", label: "My progress", icon: BarChart3 },
  ] },
  { title: "Teacher", links: [{ to: "/teacher", label: "Class overview", icon: Users }] },
  { title: "ML Lab", links: [
    { to: "/lab", label: "Models", icon: BrainCircuit, end: true },
    { to: "/lab/playground", label: "Playground", icon: FlaskConical },
    { to: "/lab/dataset", label: "Training data", icon: Database },
    { to: "/lab/benchmark", label: "OCR benchmark", icon: Gauge },
  ] },
];

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("theme") || "";
    } catch {
      return "";
    }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try {
      if (theme) localStorage.setItem("theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  const isDark = theme ? theme === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  return [isDark, () => setTheme(isDark ? "light" : "dark")];
}

export default function App() {
  const [isDark, toggleTheme] = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setMenuOpen(false), [location.pathname]);

  return (
    <StudentProvider>
      <div className="shell">
        <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
          <div className="brand">
            <div className="brand-mark">∑</div>
            <div>
              <div className="brand-name">StepWise</div>
              <div className="brand-sub">Explainable ML math tutor</div>
            </div>
          </div>
          {NAV.map((section) => (
            <nav className="nav-section" key={section.title} aria-label={section.title}>
              <div className="nav-title">{section.title}</div>
              {section.links.map(({ to, label, icon: Icon, end }) => (
                <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
                  <Icon size={17} /> {label}
                </NavLink>
              ))}
            </nav>
          ))}
          <div className="sidebar-foot">
            <div className="row" style={{ gap: 6, fontWeight: 600, color: "var(--text)" }}><Boxes size={14} /> Classical ML only</div>
            scikit-learn · OpenCV · SymPy. Runs offline on a CPU — no deep learning, no API keys.
          </div>
        </aside>

        <div className="main">
          <header className="topbar">
            <button className="btn ghost icon-btn menu-btn" onClick={() => setMenuOpen((o) => !o)} aria-label="Open menu">
              <Menu size={18} />
            </button>
            <StudentSelector />
            <button className="btn ghost icon-btn" onClick={toggleTheme} aria-label="Toggle dark mode">
              {isDark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </header>
          <main className="content">
            <ErrorBoundary key={location.pathname}>
            <Routes>
              <Route path="/" element={<UploadPage />} />
              <Route path="/analysis/:submissionId" element={<AnalysisPage />} />
              <Route path="/practice" element={<PracticePage />} />
              <Route path="/progress" element={<ProgressPage />} />
              <Route path="/teacher" element={<TeacherPage />} />
              <Route path="/lab" element={<LabPage />} />
              <Route path="/lab/playground" element={<PlaygroundPage />} />
              <Route path="/lab/dataset" element={<DatasetPage />} />
              <Route path="/lab/benchmark" element={<BenchmarkPage />} />
              <Route path="/lab/:name" element={<LabModelPage />} />
            </Routes>
            </ErrorBoundary>
          </main>
        </div>
      </div>
    </StudentProvider>
  );
}

import React, { useState } from "react";
import { Plus, UserRound } from "lucide-react";
import { useStudent } from "../api/StudentContext.jsx";

export default function StudentSelector() {
  const { students, selectedStudentId, setSelectedStudentId, createStudent } = useStudent();
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    try {
      await createStudent(name.trim());
      setName("");
      setCreating(false);
    } finally {
      setBusy(false);
    }
  };

  if (creating) {
    return (
      <form onSubmit={submit} className="row student-picker">
        <input autoFocus placeholder="New student name" className="student-select" value={name} onChange={(e) => setName(e.target.value)} />
        <button className="btn sm" disabled={busy}>Add</button>
        <button type="button" className="btn sm ghost" onClick={() => setCreating(false)}>Cancel</button>
      </form>
    );
  }
  return (
    <div className="row student-picker">
      <UserRound size={18} className="muted hide-sm" aria-hidden />
      <select aria-label="Student" className="student-select" value={selectedStudentId} onChange={(e) => setSelectedStudentId(e.target.value)}>
        <option value="" disabled>Select a student…</option>
        {students.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name}{s.grade_level ? ` · ${s.grade_level}` : ""}
          </option>
        ))}
      </select>
      <button className="btn sm secondary" onClick={() => setCreating(true)} title="Add student">
        <Plus size={15} /> <span className="hide-sm">New</span>
      </button>
    </div>
  );
}

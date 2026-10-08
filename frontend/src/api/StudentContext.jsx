import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { endpoints } from "./client";

const StudentContext = createContext(null);

const STORAGE_KEY = "ai_tutor_selected_student_id";

export function StudentProvider({ children }) {
  const [students, setStudents] = useState([]);
  const [selectedStudentId, setSelectedStudentId] = useState(
    () => localStorage_getSafe(STORAGE_KEY) || ""
  );
  const [loading, setLoading] = useState(true);

  const refreshStudents = useCallback(async () => {
    setLoading(true);
    try {
      const res = await endpoints.listStudents();
      setStudents(res.data);
      if (!selectedStudentId && res.data.length > 0) {
        setSelectedStudentId(res.data[0].id);
      }
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    refreshStudents();
  }, [refreshStudents]);

  useEffect(() => {
    if (selectedStudentId) {
      window.localStorage?.setItem(STORAGE_KEY, selectedStudentId);
    }
  }, [selectedStudentId]);

  const createStudent = useCallback(async (name, gradeLevel) => {
    const res = await endpoints.createStudent({ name, grade_level: gradeLevel || null });
    setStudents((prev) => [res.data, ...prev]);
    setSelectedStudentId(res.data.id);
    return res.data;
  }, []);

  const value = {
    students,
    loading,
    selectedStudentId,
    setSelectedStudentId,
    refreshStudents,
    createStudent,
  };

  return <StudentContext.Provider value={value}>{children}</StudentContext.Provider>;
}

function localStorage_getSafe(key) {
  try {
    return window.localStorage?.getItem(key);
  } catch {
    return null;
  }
}

export function useStudent() {
  const ctx = useContext(StudentContext);
  if (!ctx) throw new Error("useStudent must be used within StudentProvider");
  return ctx;
}

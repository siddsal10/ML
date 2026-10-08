import axios from "axios";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const api = axios.create({ baseURL: API_BASE_URL, timeout: 120000 });

export function errorMessage(err) {
  return err?.response?.data?.detail
    ? typeof err.response.data.detail === "string"
      ? err.response.data.detail
      : JSON.stringify(err.response.data.detail)
    : err?.message || "Something went wrong";
}

const multipart = { headers: { "Content-Type": "multipart/form-data" } };

export const endpoints = {
  health: () => api.get("/health"),

  listStudents: () => api.get("/api/students"),
  createStudent: (payload) => api.post("/api/students", payload),

  listSkills: () => api.get("/api/skills"),

  analyzeSubmission: (formData) => api.post("/api/submissions/analyze", formData, multipart),
  getAnalysis: (id) => api.get(`/api/submissions/${id}/analysis`),
  correctSymbols: (id, corrections) => api.post(`/api/submissions/${id}/symbols/corrections`, { corrections }),
  reanalyzeSubmission: (id, text) => api.post(`/api/submissions/${id}/reanalyze`, { corrected_transcription: text }),
  originalImageUrl: (id) => `${API_BASE_URL}/api/submissions/${id}/image`,

  getSkills: (studentId) => api.get(`/api/students/${studentId}/skills`),
  getProgress: (studentId) => api.get(`/api/students/${studentId}/progress`),
  getHistory: (studentId) => api.get(`/api/students/${studentId}/history`),

  nextPractice: (studentId, skill) => api.get(`/api/practice/${studentId}/next`, { params: skill ? { skill } : {} }),
  answerPractice: (payload) => api.post("/api/practice/answer", payload),

  classOverview: () => api.get("/api/teacher/class"),

  labModels: () => api.get("/api/lab/models"),
  labModel: (name) => api.get(`/api/lab/models/${name}`),
  labRetrain: () => api.post("/api/lab/models/symbol_classifier/retrain"),
  labJob: (id) => api.get(`/api/lab/jobs/${id}`),
  labActivate: (name, version) => api.post(`/api/lab/models/${name}/activate`, { version }),
  labPredict: (formData) => api.post("/api/lab/symbols/predict", formData, multipart),
  labDataset: () => api.get("/api/lab/dataset"),

  listProviders: () => api.get("/api/providers"),
  runBenchmark: (payload) => api.post("/api/benchmark/run", payload),
};

export default api;

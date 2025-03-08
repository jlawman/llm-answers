import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface JobState {
  jobId: string | null;
  jobStatus: 'idle' | 'running' | 'completed' | null;
  startTime: number | null;
  setJobId: (id: string | null) => void;
  setJobStatus: (status: 'idle' | 'running' | 'completed' | null) => void;
  setStartTime: (time: number | null) => void;
  resetJob: () => void;
}

export const useJobStore = create<JobState>()(
  persist(
    (set) => ({
      jobId: null,
      jobStatus: null,
      startTime: null,
      setJobId: (id) => set({ jobId: id }),
      setJobStatus: (status) => set({ jobStatus: status }),
      setStartTime: (time) => set({ startTime: time }),
      resetJob: () => set({ jobId: null, jobStatus: null, startTime: null }),
    }),
    {
      name: 'job-storage',
    }
  )
);
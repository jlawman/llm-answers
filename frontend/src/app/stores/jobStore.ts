import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface JobState {
  jobStatus: 'idle' | 'running' | 'completed';
  startTime: number | null;
  setJobStatus: (status: 'idle' | 'running' | 'completed') => void;
  setStartTime: (time: number | null) => void;
}

export const useJobStore = create<JobState>()(
  persist(
    (set) => ({
      jobStatus: 'idle',
      startTime: null,
      setJobStatus: (status) => set({ jobStatus: status }),
      setStartTime: (time) => set({ startTime: time }),
    }),
    {
      name: 'job-storage',
    }
  )
);
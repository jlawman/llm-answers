interface Job {
    id: string;
    status: 'idle' | 'running' | 'completed';
    startTime: number;
  }
  
  class JobManager {
    private jobs: Map<string, Job> = new Map();
  
    createJob(): string {
      const id = Date.now().toString();
      this.jobs.set(id, { id, status: 'idle', startTime: Date.now() });
      return id;
    }
  
    startJob(id: string): void {
      const job = this.jobs.get(id);
      if (job) {
        job.status = 'running';
        job.startTime = Date.now();
        
        // Simulate a long-running job
        setTimeout(() => {
          job.status = 'completed';
        }, 60000);
      }
    }
  
    getJobStatus(id: string): Job | undefined {
      return this.jobs.get(id);
    }
  }
  
  export const jobManager = new JobManager();
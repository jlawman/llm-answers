export async function startJob() {
    const response = await fetch('/api/long-running-function', {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to start job');
    }
    return response.json();
  }
  
  export async function getJobStatus(jobId: string) {
    const response = await fetch(`/api/long-running-function?id=${jobId}`);
    if (!response.ok) {
      throw new Error('Failed to get job status');
    }
    return response.json();
  }
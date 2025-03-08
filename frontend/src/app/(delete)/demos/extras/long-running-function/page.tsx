'use client';

import { useEffect } from 'react';
import { useJobStore } from '@/stores/jobStore';
import { startJob, getJobStatus } from '@/lib/api';

export default function LongRunningFunctionDemo() {
  const { jobId, jobStatus, setJobId, setJobStatus, setStartTime, resetJob } = useJobStore();

  useEffect(() => {
    const pollJobStatus = async () => {
      if (jobId && jobStatus === 'running') {
        try {
          const job = await getJobStatus(jobId);
          setJobStatus(job.status);
          setStartTime(job.startTime);

          if (job.status === 'running') {
            setTimeout(pollJobStatus, 5000); // Poll every 5 seconds
          }
        } catch (error) {
          console.error('Error polling job status:', error);
          resetJob();
        }
      }
    };

    pollJobStatus();
  }, [jobId, jobStatus, setJobStatus, setStartTime, resetJob]);

  const handleStartJob = async () => {
    try {
      const { jobId } = await startJob();
      setJobId(jobId);
      setJobStatus('running');
      setStartTime(Date.now());
    } catch (error) {
      console.error('Error starting job:', error);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Long-Running Function Demo</h1>
      <button
        onClick={handleStartJob}
        disabled={jobStatus === 'running'}
        className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
      >
        {jobStatus === 'running' ? 'Job Running...' : 'Start Job'}
      </button>
      {jobStatus === 'running' && (
        <div className="mt-4">
          <div className="animate-pulse bg-gray-300 h-4 w-full rounded"></div>
          <p className="mt-2">Job is running. You can navigate away and come back to check the status.</p>
        </div>
      )}
      {jobStatus === 'completed' && (
        <p className="mt-4 text-green-600">Job completed successfully!</p>
      )}
      {jobId && (
        <p className="mt-2">Job ID: {jobId}</p>
      )}
    </div>
  );
}
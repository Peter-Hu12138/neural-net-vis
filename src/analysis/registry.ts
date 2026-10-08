import { jobs as attributionJobs } from './attribution';
import { jobs as embedJobs } from './embed';
import { jobs as layerStatsJobs } from './layerStats';
import type { Job } from './protocol';
import { jobs as unitJobs } from './units';

/** Every analysis the worker can run, by name. */
export const registry: Record<string, Job> = { ...layerStatsJobs, ...unitJobs, ...attributionJobs, ...embedJobs };

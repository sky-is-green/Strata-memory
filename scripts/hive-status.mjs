#!/usr/bin/env node
/**
 * HIVE-OPS §8 - unattended status collector.
 *
 * Appends a timestamped hive snapshot to hive-status.log: worker beacons,
 * branch tip, ahead/behind vs the integration branch, staged-file count,
 * and the latest CI verdicts when `gh` is authenticated.
 *
 * Schedule it from any OS scheduler (see HIVE-OPS.md); agents read the log
 * on wake instead of pretending to watch. Safe to run concurrently-ish;
 * appends are single-pass. Fails soft: every subsystem is best-effort.
 *
 * Config via env: HIVE_INTEGRATION (default "fork/hive-studio").
 */

import { execFileSync } from 'node:child_process';
import { readFileSync, existsSync, readdirSync, appendFileSync } from 'node:fs';
import { join } from 'node:path';

function git(args) {
  try { return execFileSync('git', args, { encoding: 'utf8' }).trim(); }
  catch { return null; }
}
function gh(args) {
  try { return execFileSync('gh', args, { encoding: 'utf8' }).trim(); }
  catch { return null; }
}

const lines = [];
lines.push(`=== ${new Date().toISOString()} ===`);

// Worker beacons from the claims registry (BOM-tolerant: PowerShell trap).
const claimsDir = join(process.cwd(), '.claims');
if (existsSync(claimsDir)) {
  for (const name of readdirSync(claimsDir).filter((n) => n.endsWith('.json'))) {
    try {
      const raw = readFileSync(join(claimsDir, name), 'utf8').replace(/^\uFEFF/, '');
      const c = JSON.parse(raw);
      lines.push(
        `bee ${c.task_id ?? name} [${(c.worker ?? '?').slice(0, 40)}] ${c.beacon ?? '(no beacon)'}`
      );
    } catch (e) {
      lines.push(`bee ${name}: unreadable claim (${e.message.slice(0, 60)})`);
    }
  }
}

// Branch tip and divergence from the integration branch.
const branch = git(['branch', '--show-current']) || '(detached)';
lines.push(`branch: ${branch} @ ${git(['rev-parse', '--short', 'HEAD']) ?? '?'}`);
const integration = process.env.HIVE_INTEGRATION || 'fork/hive-studio';
const ab = git(['rev-list', '--left-right', '--count', `${branch}...${integration}`]);
lines.push(ab ? `ahead/behind ${integration}: ${ab.replace(/\s+/, ' ahead / ')} behind` : `no integration ref ${integration}`);

// Staged-file count (unclaimed-work pressure indicator).
try {
  const staged = git(['diff', '--cached', '--name-only'])
    ?.split('\n').filter(Boolean).length ?? '?';
  lines.push(`staged files: ${staged}`);
} catch { /* non-fatal */ }

// Latest CI verdicts, when gh is installed and authenticated.
const ci = gh(['run', 'list', '--limit', '2']);
if (ci) for (const l of ci.split('\n')) lines.push(`ci: ${l.slice(0, 110)}`);

appendFileSync(join(process.cwd(), 'hive-status.log'), lines.join('\n') + '\n');
console.log(lines.join('\n'));

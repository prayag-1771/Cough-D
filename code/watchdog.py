"""Watch the long running jobs and say so when one stops making progress.

    python -I watchdog.py                       (status every 30 min)
    python -I watchdog.py --once                (one report, then exit)

Three jobs run for hours here, and all three can fail in ways that look like
running: a training sweep whose process survives but whose loop has wedged, an
ingest whose worker pool has died leaving the parent alive, a download that has
stopped receiving bytes. A process table check alone does not catch any of
them, so progress is checked rather than liveness.

The signal used is the job's own log file. Every one of these jobs prints at
least once a minute while healthy, so a log whose size has not changed across
consecutive checks is the evidence that the job has stopped working, whatever
the process table says. Each job also has a completion marker, so a finished
job is not reported as a dead one.

Checks run every CHECK_EVERY seconds. A healthy report is printed only every
REPORT_EVERY seconds, to keep the log readable, but a problem is printed the
moment it is seen.
"""
import argparse
import glob
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')

CHECK_EVERY = 600          # look this often
REPORT_EVERY = 1800        # print a healthy report this often
STALL_CHECKS = 2           # consecutive unchanged checks before calling a stall

# name, log file, process command line fragment, completion marker, progress probe
JOBS = [
    ('sweep', 'sweep*.out|resume.out', 'sweep.py', 'SWEEP COMPLETE',
     ('csv', os.path.join(ROOT, 'runs_coswara'), '.csv', 66)),
    ('coughvid', 'coughvid*.out', 'coughvid_ingest', 'imaged ',
     ('png', os.path.join(ROOT, 'data', 'coughvid_img', 'linear'), '.png', 8435)),
    ('fsd50k_dl', 'fsd50k_dl*.out', 'fsd50k_ingest', 'download only',
     ('bytes', os.path.join(ROOT, 'data', 'fsd50k_raw'), '', 0)),
]


def newest_log(pattern):
    """The most recently written log matching the pattern, or None.

    A job that has to be relaunched writes to a new log, so the watcher
    resolves the pattern each time rather than holding a fixed name. Watching
    a stale log would report a healthy job as stalled, which is exactly what
    happened when the grid was restarted under a name the pattern did not
    cover: the count of finished runs kept climbing while the watcher reported
    a stall against a log nothing was writing to any more. Several patterns
    separated by a bar are therefore accepted, and the newest file across all
    of them wins.
    """
    paths = []
    for part in pattern.split('|'):
        paths.extend(glob.glob(os.path.join(LOGS, part)))
    return max(paths, key=os.path.getmtime) if paths else None


def running(fragment):
    """Is a python process running with this fragment in its command line?

    Restricted to interpreter processes so the probe cannot match itself: the
    query carries the fragment in its own command line, and an unrestricted
    enumeration of the process table counts the query.
    """
    ps = ("Get-CimInstance Win32_Process -Filter \"Name LIKE '%%python%%'\" | "
          "Where-Object { $_.CommandLine -like '*%s*' -and "
          "$_.ProcessId -ne %d } | Measure-Object | "
          "Select-Object -ExpandProperty Count" % (fragment, os.getpid()))
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                           capture_output=True, text=True, timeout=180,
                           stdin=subprocess.DEVNULL)
        return int((r.stdout or '').strip() or '0') > 0
    except Exception:
        return True            # unknown counts as running, never as dead


def progress(probe):
    """Count or size the job's output, walking subdirectories.

    The image sets put their files one level down, under the class name, so a
    flat listing of the condition directory finds only the two class folders
    and reports zero while the job is working correctly.
    """
    kind, path, ext, _target = probe
    try:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                if kind == 'bytes':
                    total += os.path.getsize(os.path.join(root, f))
                elif f.endswith(ext):
                    total += 1
        return total
    except Exception:
        return 0


def pretty(probe, value):
    kind, _path, _ext, target = probe
    if kind == 'bytes':
        return '%.2f GB' % (value / 2 ** 30)
    return '%d%s' % (value, '/%d' % target if target else '')


def say(msg):
    print('%s  %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def check(state, report):
    """One pass. Returns the number of problems seen."""
    problems = 0
    lines = []
    for name, pattern, fragment, marker, probe in JOBS:
        log = newest_log(pattern)
        size = os.path.getsize(log) if log else -1
        try:
            done = marker in open(log, encoding='utf-8', errors='replace').read()
        except Exception:
            done = False
        if log is None:
            done = False
        value = progress(probe)
        prev = state.get(name, {})

        if done:
            status = 'DONE'
            state[name] = {'size': size, 'value': value, 'stale': 0,
                           'done': True}
        elif prev.get('done'):
            status = 'DONE'
            state[name] = prev
        else:
            stale = prev.get('stale', 0)
            stale = stale + 1 if size == prev.get('size') else 0
            alive = running(fragment)
            state[name] = {'size': size, 'value': value, 'stale': stale,
                           'done': False}
            if not alive:
                status = 'EXITED without finishing'
                problems += 1
            elif stale >= STALL_CHECKS:
                status = ('STALLED: log unchanged for %d checks'
                          % (stale + 1))
                problems += 1
            else:
                status = 'ok'
        lines.append('    %-11s %-34s %s'
                     % (name, status, pretty(probe, value)))

    if problems or report:
        say('status%s' % ('  PROBLEM' if problems else ''))
        for ln in lines:
            print(ln, flush=True)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--check-every', type=int, default=CHECK_EVERY)
    ap.add_argument('--report-every', type=int, default=REPORT_EVERY)
    a = ap.parse_args()

    say('watching %d jobs; checking every %d s, reporting every %d s'
        % (len(JOBS), a.check_every, a.report_every))
    state = {}
    if a.once:
        check(state, True)
        return

    last_report = 0.0
    while True:
        due = (time.time() - last_report) >= a.report_every
        if check(state, due) or due:
            last_report = time.time()
        if all(state.get(n, {}).get('done') for n, _l, _f, _m, _p in JOBS):
            say('every job reports finished; nothing left to watch')
            return
        time.sleep(a.check_every)


if __name__ == '__main__':
    main()

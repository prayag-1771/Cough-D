"""Hold this project's heavy work until the prioritised project has finished.

    python -I queue_runner.py                 (watch, then start the lanes)
    python -I queue_runner.py --status        (report and exit)
    python -I queue_runner.py --lane gpu     (run one lane in the foreground)

Another project on this machine has priority. It runs in bursts with gaps
between its stages, so a single look at the process table is not evidence that
it has finished: an earlier check during one of those gaps saw nothing and
started work that then had to be stood down. This waits for a sustained quiet
period instead, and only then starts, which is the whole reason the file
exists.

Work is divided into two lanes by the resource each job is limited by. The
lanes run at the same time as one another and sequentially within themselves:
training is bound by the graphics card and the ingests are bound by the
processor and the network, so a lane of each uses the machine fully, whereas
two ingests at once would merely divide the same cores.

Everything runs at below normal priority, so if the prioritised project comes
back the operating system hands it the processor ahead of this without
anything needing to intervene. Every job is individually resumable, so being
stopped at any point costs only the run in progress.
"""
import argparse
import os
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
LOGS = os.path.join(ROOT, 'logs')
D = os.path.join('..', 'data')

# The prioritised project, matched on its path in the command line.
MARKER = 'B_i_g'

LANES = {
    'gpu': [
        ('sweep_coswara',
         ['-u', '-I', 'sweep.py',
          '--images', os.path.join(D, 'coswara_img'),
          '--out', os.path.join('..', 'runs_coswara'),
          '--weights', os.path.join('..', 'weights'),
          '--plan', 'all']),
    ],
    'cpu': [
        # VocalSound first: it is a tenth of the size and it is the set that
        # tests whether the Coswara accuracy survives hard negatives, which is
        # the open question.
        ('vocalsound',
         ['-u', '-I', 'vocalsound_ingest.py', '--download',
          '--out', os.path.join(D, 'vocalsound_img'),
          '--split', 'te', '--workers', '6']),
        ('coughvid',
         ['-u', '-I', 'coughvid_ingest.py',
          '--out', os.path.join(D, 'coughvid_img'),
          '--match', 'coswara', '--workers', '6']),
    ],
}

# Return the matching process ids, not a count, so what matched can be logged.
#
# The Name filter is not an optimisation, it is the fix for a real bug. The
# probe is itself a powershell process whose command line contains the marker
# string, so an unfiltered enumeration of Win32_Process matched the query that
# was doing the matching and returned 1 forever: the queue reported the other
# project busy for as long as it ran and would never have started. Restricting
# to interpreter processes excludes the probe by construction, and the
# runner's own pid is excluded as well.
PS_PIDS = (
    "Get-CimInstance Win32_Process -Filter "
    "\"Name LIKE '%%python%%' OR Name LIKE '%%matlab%%'\" | "
    "Where-Object { $_.CommandLine -like '*%s*' -and $_.ProcessId -ne %d } | "
    "Select-Object -ExpandProperty ProcessId"
)

DETACHED = 0
if os.name == 'nt':
    # CREATE_NO_WINDOW, not DETACHED_PROCESS. A detached process cannot
    # duplicate handles to children, so any job that builds a process pool
    # dies inside multiprocessing with "WinError 5 Access is denied" from
    # DuplicateHandle. That is what killed the first ingest run after three
    # minutes. CREATE_NO_WINDOW gives the same absence of a console window
    # while leaving handle inheritance intact, and the job still outlives this
    # process because nothing waits on it.
    DETACHED = (0x08000000 |        # CREATE_NO_WINDOW
                0x00000200 |        # CREATE_NEW_PROCESS_GROUP
                0x00004000)         # BELOW_NORMAL_PRIORITY_CLASS


def marker_pids():
    """Process ids of the prioritised project. None means the check failed."""
    cmd = PS_PIDS % (MARKER, os.getpid())
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command', cmd],
                           capture_output=True, text=True, timeout=180,
                           stdin=subprocess.DEVNULL)
        return [int(ln) for ln in (r.stdout or '').split() if ln.strip().isdigit()]
    except Exception as exc:
        say('check failed (%s)' % exc.__class__.__name__)
        return None


def busy():
    """Is the prioritised project running? A failed check counts as busy."""
    pids = marker_pids()
    return True if pids is None else bool(pids)


def say(msg):
    print('%s  %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def logfile(name, mode='ab'):
    os.makedirs(LOGS, exist_ok=True)
    return open(os.path.join(LOGS, name), mode, buffering=0)


def run_lane(lane):
    """Run one lane's jobs in order, in this process, logging each."""
    for name, argv in LANES[lane]:
        say('lane %s: starting %s' % (lane, name))
        out, err = logfile(name + '.out'), logfile(name + '.err')
        try:
            code = subprocess.call([sys.executable] + argv, cwd=CODE,
                                   stdout=out, stderr=err,
                                   stdin=subprocess.DEVNULL)
        finally:
            out.close()
            err.close()
        say('lane %s: %s finished with code %s' % (lane, name, code))
    say('lane %s: all jobs done' % lane)


def start_lanes():
    for lane in LANES:
        out = logfile('lane_%s.out' % lane)
        p = subprocess.Popen(
            [sys.executable, '-u', '-I', os.path.join(CODE, 'queue_runner.py'),
             '--lane', lane],
            cwd=CODE, stdout=out, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=DETACHED, close_fds=True)
        say('started lane %-3s PID %d  -> logs/lane_%s.out'
            % (lane, p.pid, lane))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clear-minutes', type=int, default=15,
                    help='consecutive quiet minutes required before starting')
    ap.add_argument('--poll', type=int, default=120, help='seconds between checks')
    ap.add_argument('--max-hours', type=float, default=24.0)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--lane', default=None, choices=sorted(LANES))
    a = ap.parse_args()

    if a.status:
        pids = marker_pids()
        if pids is None:
            print('%s: the check FAILED, so it would be treated as busy' % MARKER)
        elif pids:
            print('%s is RUNNING: %s' % (MARKER, ', '.join(str(p) for p in pids)))
        else:
            print('%s is not running' % MARKER)
        for lane, jobs in sorted(LANES.items()):
            print('  lane %-3s: %s' % (lane, ', '.join(n for n, _ in jobs)))
        return

    if a.lane:
        run_lane(a.lane)
        return

    need = max(1, (a.clear_minutes * 60) // a.poll)
    say('waiting for %s to finish; need %d consecutive quiet checks (%d min)'
        % (MARKER, need, a.clear_minutes))

    # Report what the check sees before trusting it. A previous version
    # matched its own probe process and so reported the other project busy
    # for as long as it ran, logging nothing the whole time; an unexplained
    # silence must never again be mistaken for patience.
    first = marker_pids()
    if first is None:
        say('WARNING: the check itself failed; will keep retrying and will '
            'not start while it is failing')
    else:
        say('check sees %d %s process(es)%s' % (len(first), MARKER,
            (': ' + ', '.join(str(p) for p in first)) if first else ''))

    clear, polls = 0, 0
    deadline = time.time() + a.max_hours * 3600
    while time.time() < deadline:
        pids = marker_pids()
        polls += 1
        if pids is None or pids:
            if clear:
                say('%s active again, quiet streak reset (was %d/%d)'
                    % (MARKER, clear, need))
            elif polls <= 3 or polls % 10 == 0:
                say('still busy (%s), waiting'
                    % ('check failing' if pids is None
                       else '%d process(es)' % len(pids)))
            clear = 0
        else:
            clear += 1
            # The first few are logged unconditionally so that progress is
            # visible immediately rather than only near the threshold.
            if clear <= 3 or clear % 5 == 0 or clear == need:
                say('quiet %d/%d' % (clear, need))
            if clear >= need:
                break
        time.sleep(a.poll)
    else:
        say('gave up after %.1f h without a quiet period' % a.max_hours)
        return

    say('machine is free, starting %d lane(s)' % len(LANES))
    start_lanes()
    say('lanes started; this runner is done')


if __name__ == '__main__':
    main()

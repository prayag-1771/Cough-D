"""Run the remaining work to completion, in order, without supervision.

    python -I finish.py

The grid over the prompt derived corpus is already running when this starts.
Everything after it is sequential because there is one card, and each stage
needs the stage before it: the classifier derived grid cannot share the device
with the grid already running without both of them overrunning its memory, the
external evaluation needs the weights the grids retain, and the tables and
figures need the decision files all of those produce.

Each stage is skipped when its output is already present, so this can be
restarted after an interruption without repeating work that survived. The
machine running this has rebooted mid grid once already.

The classifier derived grid writes its weights to a separate directory. Both
grids name a run by architecture, condition and seed, so a shared directory
would have the second corpus silently overwrite the first corpus' weights and
the external evaluation would then be reporting on a mixture of the two.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

COUGHVID_RUNS = os.path.join(ROOT, 'runs_coughvid')
COUGHVID_JOBS = 20


def say(msg):
    print('%s  %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def running(fragment):
    """Is a python process other than this one running that script?"""
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
        return True          # unknown counts as running, never as finished


def wait_for(fragment, label):
    """Block until no process matching the fragment is left."""
    if not running(fragment):
        say('%s is not running' % label)
        return
    say('waiting for %s to finish' % label)
    quiet = 0
    while True:
        time.sleep(120)
        if running(fragment):
            quiet = 0
            continue
        quiet += 1
        # Two consecutive quiet checks, so a momentary miss does not start
        # the next stage on top of a grid that is still going.
        if quiet >= 2:
            say('%s has finished' % label)
            return


def run(args, label, cwd=HERE):
    say('start: %s' % label)
    r = subprocess.run([PY, '-u', '-I'] + args, cwd=cwd)
    say('done:  %s (exit %d)' % (label, r.returncode))
    return r.returncode


def csv_count(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.csv')])


def main():
    wait_for('sweep.py', 'the grid over the prompt derived corpus')

    if csv_count(COUGHVID_RUNS) < COUGHVID_JOBS:
        run(['sweep.py',
             '--images', os.path.join('..', 'data', 'coughvid_img'),
             '--out', os.path.join('..', 'runs_coughvid'),
             '--weights', os.path.join('..', 'weights_coughvid'),
             '--plan', 'core'],
            'the grid over the classifier derived corpus')
    else:
        say('the classifier derived grid is already complete')

    run(['external_eval.py',
         '--weights', os.path.join('..', 'weights'),
         '--images', os.path.join('..', 'data', 'fsd50k_img'),
         '--out', os.path.join('..', 'runs_fsd50k')],
        'external evaluation of the retained weights')

    run(['tables.py'], 'result tables')
    run(['threshold.py'], 'discrimination against calibration')
    run(['figures.py', '--what', 'transfer'], 'the transfer figure')
    run(['assemble.py', '--check'], 'assembling and checking the manuscript',
        cwd=ROOT)

    say('all remaining stages have run')


if __name__ == '__main__':
    main()

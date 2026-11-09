"""Restart the prompt derived grid once the classifier derived one is done.

    python -I resume_coswara.py

The two grids cannot share the card. Either one on its own fits in the four
gigabytes available; together they do not, and the driver would satisfy the
excess from host memory and run both at a fraction of the speed without
reporting anything wrong.

The order was chosen deliberately. The manuscript needs the classifier derived
results before its last section can be written, while the architectures left
in the prompt derived grid only add rows to tables that regenerate themselves.
Running the shorter grid first therefore finishes the writing several hours
earlier without changing when the computing finishes.

Nothing already computed is at risk: the sweep skips any run whose decision
file exists, so restarting it picks up exactly the runs that are missing.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COUGHVID = os.path.join(ROOT, 'runs_coughvid')
WANT = 20


def say(msg):
    print('%s  %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def csvs(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.csv')])


def sweep_running():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name LIKE '%python%'\" | "
          "Where-Object { $_.CommandLine -like '*sweep.py*' } | "
          "Measure-Object | Select-Object -ExpandProperty Count")
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                           capture_output=True, text=True, timeout=180,
                           stdin=subprocess.DEVNULL)
        return int((r.stdout or '').strip() or '0') > 0
    except Exception:
        return True


def main():
    say('waiting for the classifier derived grid to reach %d runs' % WANT)
    started = time.time()
    while True:
        done = csvs(COUGHVID)
        if done >= WANT:
            say('classifier derived grid complete at %d runs' % done)
            break
        # If that grid has stopped without finishing, do not wait for ever;
        # the remaining architectures are worth more than a stalled queue.
        if done > 0 and not sweep_running():
            say('classifier derived grid is not running and sits at %d of %d; '
                'continuing anyway' % (done, WANT))
            break
        if time.time() - started > 6 * 3600:
            say('waited six hours; continuing anyway')
            break
        time.sleep(120)

    # Let any evaluation started by the chain release the card first.
    time.sleep(90)
    say('restarting the prompt derived grid for the architectures left')
    r = subprocess.run(
        [sys.executable, '-u', '-I', 'sweep.py',
         '--images', os.path.join('..', 'data', 'coswara_img'),
         '--out', os.path.join('..', 'runs_coswara'),
         '--weights', os.path.join('..', 'weights'),
         '--plan', 'rest'], cwd=HERE)
    say('prompt derived grid exited with %d' % r.returncode)

    for script, label in (('external_eval.py', 'external evaluation'),
                          ('tables.py', 'result tables'),
                          ('threshold.py', 'discrimination against calibration'),
                          ('error_analysis.py', 'false positives by class'),
                          ('trace_numbers.py', 'tracing quoted numbers')):
        args = [sys.executable, '-u', '-I', script]
        if script == 'external_eval.py':
            args += ['--weights', os.path.join('..', 'weights'),
                     '--images', os.path.join('..', 'data', 'fsd50k_img'),
                     '--out', os.path.join('..', 'runs_fsd50k')]
        say('start: %s' % label)
        subprocess.run(args, cwd=HERE)

    for venue in ('ieee', 'springer'):
        say('assembling the %s target' % venue)
        subprocess.run([sys.executable, '-u', '-I', 'assemble.py',
                        '--venue', venue, '--check'], cwd=ROOT)
    say('everything has run')


if __name__ == '__main__':
    main()

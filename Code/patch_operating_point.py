
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] check | Patch exp_operating_point.py to use independent clean samples
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
[P] Patch exp_operating_point.py: estimate the clean quantile from
independent samples

A pre-submission check found two errors in the alarm-budget conversion, both
in the same direction, both making the result look better than it is.

  Error one   Validation windows are not independent
              The window is 30 frames at stride one, so adjacent windows
              share 29 of 30 frames. The 8,731 held-out windows are not 8,731
              independent samples. Estimating a (1-FPR) quantile needs
              independent samples, so the inference that 8,731 supports a
              strictest budget of 1.8 alarms per hour does not hold.

  Error two   The [D1] judgement was backwards
              The original comment read: if deployment uses overlapping
              windows the decision count multiplies by the stride, so
              counting non-overlapping windows is conservative. Deployment
              being worse means the smaller number is optimistic, not
              conservative: N times more decisions needs an N times lower
              per-window false-positive rate, hence a higher threshold and a
              larger floor.

This patch changes only how clean scores are sampled. Training, injection,
features and the model are untouched.

-- Three design decisions ----------------------------------------

  [D1] Full independence: keep one window in every LOOKBACK
       At stride one with a 30-frame window, only every 30th window is
       non-overlapping. That is the correct treatment for a tail quantile,
       at the cost of dropping from 8,731 samples to about 290.
       A block bootstrap would give an effective sample size between the two,
       but needs half a page of method; full independence is its conservative
       end.

  [D2] Report both decision frequencies, conclude from the non-overlapping one
       15,850 per hour corresponds to scoring only non-overlapping windows.
       475,200 per hour corresponds to scoring every frame.
       The paper reports the former and states in the text that the latter
       tightens the budget by a factor of 30.

  [D3] No extrapolation
       The original fell back to a generalised-Pareto tail fit when samples
       ran short. That extrapolation produced inconsistent results, with
       tighter budgets yielding lower thresholds, so anything unestimable
       returns none here.

Usage: python3 patch_operating_point.py      # patch, keeping a backup
       python3 exp_operating_point.py        # re-run
"""
import os, re, sys, shutil

TARGET = 'exp_operating_point.py'

NEW_FUNC = '''
# ══════════════════════════════════════════════════════════════════
#  [P] Quantile estimation from independent clean samples (pre-submission fix)
#      See the patch notes at the top; the original is kept as _tpr_at_fpr_legacy
# ══════════════════════════════════════════════════════════════════
INDEP_STRIDE = 30          # [D1] 30-frame window at stride 1: every 30th is disjoint
WPH_NONOVERLAP = 15850.0   # [D2] 3600 / 0.227, non-overlapping windows only
WPH_SLIDING = 475200.0     # [D2] 3600 / (1 frame at 132 Hz), every frame scored


def tpr_at_fpr(scores, y, fpr, tail_fit=False):
    """
    Set the threshold from the quantile of *independent* clean scores.
    Returns (TPR, threshold, extrapolated).

    The only change from the original: clean scores are thinned by
    INDEP_STRIDE before the quantile. Overlapping windows are not independent,
    and using them overstates the effective sample size, which then claims an

    unestimable budget is reachable.

    [D3] tail_fit is off by default. The original extrapolated with a
    generalised Pareto fit, which gave tighter budgets lower thresholds.
    """
    clean_all = scores[y < 0.5]
    atk = scores[y > 0.5]
    if len(clean_all) == 0 or len(atk) == 0:
        return np.nan, np.nan, False

    clean = clean_all[::INDEP_STRIDE]          # [D1] thin to non-overlapping
    n = len(clean)
    if fpr * n < 1.0:
        return np.nan, np.nan, True            # too few samples, no extrapolation
    thr = np.quantile(clean, 1 - fpr)
    return float((atk > thr).mean()), float(thr), False


def strictest_budget(scores, y, windows_per_hour=WPH_NONOVERLAP):
    """The strictest directly estimable budget, in alarms per hour"""
    n = len(scores[y < 0.5][::INDEP_STRIDE])
    return windows_per_hour / n if n else float('inf')


def _tpr_at_fpr_legacy(scores, y, fpr, tail_fit=True):
'''


def main():
    if not os.path.exists(TARGET):
        print(f"{TARGET} not found; run this from the script directory"); sys.exit(1)

    src = open(TARGET, encoding='utf-8').read()
    if 'INDEP_STRIDE' in src:
        print("  already patched, nothing to do"); sys.exit(0)

    bak = TARGET + '.orig'
    if not os.path.exists(bak):
        shutil.copy(TARGET, bak)
        print(f"  backed up the original to {bak}")

    anchor = 'def tpr_at_fpr(scores, y, fpr, tail_fit=True):'
    if anchor not in src:
        print(f"  anchor {anchor!r} not found; the original may have changed"); sys.exit(1)

    src = src.replace(anchor, NEW_FUNC.strip() + '\n', 1)

    # Prepend a note so anyone re-running sees that the convention changed
    note = '''# ══════════════════════════════════════════════════════════════════
#  [P] Patched by patch_operating_point.py (pre-submission check)
#      Clean quantiles now use every 30th window; no tail extrapolation.
#      The strictest estimable budget therefore moves from 1.8 to about 38/hour.
#      The original is preserved as exp_operating_point.py.orig
# ══════════════════════════════════════════════════════════════════
'''
    src = note + src
    open(TARGET, 'w', encoding='utf-8').write(src)
    print(f"  {TARGET} patched")
    print(f"""
  What changed
    clean scores thinned by 30      overlapping windows are not independent
    tail extrapolation off          it gave inconsistent results
    added strictest_budget()        reports the strictest estimable budget

  What did not change
    training, injection, features, the model, and the attack scores

  Re-run
    python3 {TARGET}

  Expected
    The two-alarms-per-hour row becomes unestimable, since only about 290
    independent samples remain. The strictest reportable budget lands near
    38/hour. Ramp and noise having no operating point at any budget is unaffected.
""")


if __name__ == '__main__':
    main()

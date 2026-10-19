"""Verifier completion guards; never evaluate or rewrite comparisons.

Pytest calls this hook after a comparison in an assertion has already failed.
Large difflib reports can monopolize workers long after the tested program exits.
This module is mounted only in verifier containers, never Solver environments.
Retry timer repair preserves the timeout specified by each original test.
"""
import pytest

_original_formatter = None
_installed_formatter = None

_TEXT = (str, bytes, bytearray)
_CONTAINERS = (list, tuple, dict, set, frozenset)


def _arm_retry_timeout(item):
    """Rearm the unchanged whole-test budget on an individual test retry.

    pytest-rerunfailures calls the internal runtestprotocol repeatedly inside
    one pytest_runtest_protocol hook. pytest-timeout's outer hook therefore
    starts its timer only once. After an initial exception/timeout that timer
    has been cancelled/consumed. Do not change the first attempt, func_only
    timers (already armed per call), disabled timers, or the retry count.
    """
    attempt = getattr(item, 'execution_count', 0)
    if attempt <= 1 or getattr(item, '_programbench_timed_attempt', None) == attempt:
        return
    import pytest_timeout
    settings = pytest_timeout._get_item_settings(item)
    if not settings.timeout or settings.timeout <= 0 or settings.func_only:
        return
    hooks = item.config.pluginmanager.hook
    hooks.pytest_timeout_cancel_timer(item=item)
    hooks.pytest_timeout_set_timer(item=item, settings=settings)
    item._programbench_timed_attempt = attempt


def _whole_signal_timer(item):
    try:
        import pytest_timeout
    except ImportError:
        return False
    settings = pytest_timeout._get_item_settings(item)
    return bool(settings.timeout and settings.timeout > 0 and not settings.func_only
                and settings.method == 'signal')


def _resume_execution_timer(item):
    """Only execution phases consume the original cumulative whole-test budget."""
    saved = getattr(item, '_programbench_paused_timer', None)
    item._programbench_paused_timer = None
    if saved and _whole_signal_timer(item):
        import signal
        seconds, interval, handler = saved
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds, interval)


def _pause_execution_timer(item, outcome):
    if not _whole_signal_timer(item):
        return
    import signal
    # This hook finishes inside pytest's protected CallInfo invocation. A
    # deadline expiring here is attributed to the real execution phase,
    # never an uncaught exception in JUnit/xdist reporting. Preserve remaining
    # execution budget across phases, not a fresh budget for each phase.
    handler = signal.getsignal(signal.SIGALRM)
    seconds, interval = signal.setitimer(signal.ITIMER_REAL, 0)
    item._programbench_paused_timer = ((seconds, interval, handler)
        if seconds > 0 and outcome.excinfo is None else None)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    item._programbench_paused_timer = None
    _arm_retry_timeout(item)
    outcome = yield
    _pause_execution_timer(item, outcome)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    _resume_execution_timer(item)
    outcome = yield
    _pause_execution_timer(item, outcome)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    _resume_execution_timer(item)
    outcome = yield
    _pause_execution_timer(item, outcome)
    item._programbench_paused_timer = None


def _oversized(value, budget, seen):
    kind = type(value)
    if kind in _TEXT:
        return len(value) > 4096
    if kind not in _CONTAINERS:
        return False
    if len(value) > 128:
        return True
    if id(value) in seen:
        return False
    seen.add(id(value))
    items = value.items() if kind is dict else value
    for item in items:
        budget[0] -= 1
        if budget[0] < 0 or _oversized(item, budget, seen):
            return True
    return False


def _description(value):
    kind = type(value)
    if kind in _TEXT + _CONTAINERS:
        return f'{kind.__name__}, length={len(value)}'
    return kind.__name__


@pytest.hookimpl(tryfirst=True)
def pytest_assertrepr_compare(config, op, left, right):
    if not (_oversized(left, [128], set()) or _oversized(right, [128], set())):
        return None
    return [
        f'comparison {op} failed (large-operand diagnostic summary)',
        'Full difflib expansion omitted by evaluator reporter; comparison outcome is unchanged.',
        'left: ' + _description(left),
        'right: ' + _description(right),
    ]


def pytest_configure(config):
    # Whole-project scoring needs every declared test. Some generated run.sh
    # files set -x/--maxfail, stopping after one genuine failure and leaving
    # most of the denominator unexecuted. Disable only early suite termination;
    # preserve test bodies, failures, per-test timeouts and retry counts.
    if config is not None:
        config.option.maxfail = 0
    # The public comparison hook is NOT first-result: pytest executes the
    # built-in formatter even if a tryfirst hook already returned a summary.
    # Bound the actual formatter entry as well, after comparison evaluation.
    from _pytest.assertion import util
    global _original_formatter, _installed_formatter
    if getattr(util.assertrepr_compare, '_programbench_bounded', False):
        return
    original = util.assertrepr_compare
    def bounded(config, op, left, right, *args, **kwargs):
        summary = pytest_assertrepr_compare(config, op, left, right)
        if summary is not None:
            return summary
        return original(config, op, left, right, *args, **kwargs)
    bounded._programbench_bounded = True
    _original_formatter = original
    _installed_formatter = bounded
    util.assertrepr_compare = bounded
    import sys
    print("PROGRAMBENCH_BOUNDED_REPORTER_ACTIVE_v1 maxfail=0", file=sys.stderr, flush=True)


def pytest_unconfigure(config):
    from _pytest.assertion import util
    if _installed_formatter is not None and util.assertrepr_compare is _installed_formatter:
        util.assertrepr_compare = _original_formatter

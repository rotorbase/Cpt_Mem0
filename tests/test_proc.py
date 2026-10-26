"""The children of the program.

**A program that outlives its app takes the card with it.** The
speech recognition and the model stay on the graphics card when the
window is closed with the cross, because the daemon is killed
without a goodbye.

Windows has the answer: a job object with the flag "kill on close".
The handle lives in the daemon; when the daemon dies, the handle is
closed, and the job kills everything in it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


class TestNoConsole:
    """No black window."""

    def test_the_flags_are_there_on_windows(self):
        from cpt_mem0 import proc

        if not proc.IS_WINDOWS:
            assert proc.no_window() == {}
            return
        assert proc.no_window()["creationflags"] == (
            subprocess.CREATE_NO_WINDOW)

    def test_a_program_runs_without_a_window(self):
        from cpt_mem0 import proc

        result = proc.run([sys.executable, "-c", "print('hallo')"],
                           capture_output=True, text=True, timeout=60)
        assert "hallo" in result.stdout


class TestDieWithMe:
    """The children die with the program."""

    def test_it_does_nothing_without_windows(self, monkeypatch):
        from cpt_mem0 import proc

        monkeypatch.setattr(proc, "IS_WINDOWS", False)
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            proc.die_with_me(process)
            assert process.poll() is None
        finally:
            process.kill()
            process.wait(timeout=10)

    @pytest.mark.skipif(sys.platform != "win32",
                        reason="the job object is a thing of Windows")
    def test_the_child_is_bound_to_a_job(self):
        """The job carries the flag "kill on close"."""
        import win32job

        from cpt_mem0 import proc

        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            before = len(proc._JOBS)
            proc.die_with_me(process)
            assert len(proc._JOBS) == before + 1

            job = proc._JOBS[-1]
            limits = win32job.QueryInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation)
            flags = limits["BasicLimitInformation"]["LimitFlags"]
            assert flags & win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            # **And the child is in the job.**
            pids = win32job.QueryInformationJobObject(
                job, win32job.JobObjectBasicProcessIdList)
            assert process.pid in pids
        finally:
            process.kill()
            process.wait(timeout=10)

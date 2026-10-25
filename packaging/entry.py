"""The entry point of the packaged app.

The .exe does one thing: it starts the daemon and opens the window.
The arguments are passed through, so ``Cpt_Mem0.exe --suspend`` starts
the daemon with the models off.

**Why not the CLI?** The CLI has many commands. The window is what
the user starts. The commands are still there for the script and for
the autostart task.
"""

import multiprocessing
import sys


def main() -> int:
    # The frozen app must not spawn a second copy through the default
    # start method - the daemon runs in a thread.
    multiprocessing.freeze_support()

    from cpt_mem0.cli import main as cli_main

    arguments = list(sys.argv[1:])
    if not arguments:
        arguments = ["ui"]
    # A bare flag belongs to the window - the CLI maps it.
    return cli_main(arguments)


if __name__ == "__main__":
    sys.exit(main())

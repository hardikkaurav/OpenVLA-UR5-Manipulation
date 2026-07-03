"""
Pause Manager
=============
Provides thread-safe synchronization for pausing, resuming, and quitting
the UR5 data collection pipeline during execution.

Runs a background daemon thread listening to sys.stdin for user commands:
    - 'p' / 'pause' : Request experiment pause at the next waypoint.
    - 'r' / 'resume': Resume paused experiment.
    - 'q' / 'quit'  : Safely terminate the experiment.
"""

import logging
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class QuitRequestedException(Exception):
    """Raised when the operator requests clean termination of the experiment."""
    pass


class PauseManager:
    """
    Manages pause, resume, and quit states across threads without polling or
    blocking the main robot control loop.
    """

    def __init__(self):
        self._pause_event = threading.Event()
        self._resume_event = threading.Event()
        self._quit_event = threading.Event()
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the stdin listener background daemon thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._pause_event.clear()
            self._resume_event.clear()
            self._quit_event.clear()
            self._listener_thread = threading.Thread(
                target=self._listen_stdin,
                name="PauseManagerStdinListener",
                daemon=True,
            )
            self._listener_thread.start()
            logger.info("PauseManager input listener started (Type P to pause, Q to quit).")

    def stop(self) -> None:
        """Stop the background listener thread."""
        with self._lock:
            self._running = False
            self._quit_event.set()
            self._resume_event.set()

    def _listen_stdin(self) -> None:
        """Background loop reading standard input for control commands."""
        while self._running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd in ("p", "pause"):
                    if not self._pause_event.is_set():
                        logger.info("Pause requested by user.")
                        self._pause_event.set()
                        self._resume_event.clear()
                elif cmd in ("r", "resume"):
                    if self._pause_event.is_set():
                        logger.info("Resume requested by user.")
                        self._resume_event.set()
                elif cmd in ("q", "quit", "exit"):
                    logger.info("Quit requested by user.")
                    self._quit_event.set()
                    self._resume_event.set()  # Wake up if waiting in pause
                    break
            except Exception as e:
                if self._running:
                    logger.debug("Error reading stdin in PauseManager: %s", e)
                break

    def check_pause(self) -> None:
        """
        Check if quit or pause has been requested.
        
        Must be called from the main control thread at clean waypoint boundaries.
        If quit is requested, raises QuitRequestedException.
        If pause is requested, blocks the main thread cleanly until resume or quit is signaled.
        """
        if self._quit_event.is_set():
            raise QuitRequestedException("Operator requested experiment termination.")

        if self._pause_event.is_set():
            print(
                "\n═══════════════════════════════════════════════════════════\n"
                "  Experiment Paused\n"
                "  Press R (or type resume) and ENTER to Resume\n"
                "  Press Q (or type quit) and ENTER to Quit\n"
                "═══════════════════════════════════════════════════════════\n",
                flush=True
            )
            logger.info("Experiment paused. Waiting for operator command (R to resume, Q to quit)...")
            
            # Wait cleanly until resume or quit is signaled
            while self._pause_event.is_set() and not self._quit_event.is_set():
                # Wait for resume event to be set by the stdin thread
                self._resume_event.wait(timeout=0.5)
                if self._resume_event.is_set() or self._quit_event.is_set():
                    break

            if self._quit_event.is_set():
                raise QuitRequestedException("Operator requested experiment termination while paused.")

            with self._lock:
                self._pause_event.clear()
                self._resume_event.clear()
            
            print(
                "\n═══════════════════════════════════════════════════════════\n"
                "  Resuming Experiment...\n"
                "═══════════════════════════════════════════════════════════\n",
                flush=True
            )
            logger.info("Experiment resumed.")

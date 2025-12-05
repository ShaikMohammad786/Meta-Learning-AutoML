import json
import time
import random
from datetime import datetime
from pathlib import Path

from constants import USERS_FOLDER


class TrainingStatusTracker:
    """
    Lightweight file-based tracker so the frontend can poll live training state.
    We append every update into history and keep the latest event under `current`.
    """

    def __init__(self, user_id: str, dataset_name: str):
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.status_dir = Path(USERS_FOLDER) / user_id / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.status_dir / f"{dataset_name}.json"

    def _retry_io(self, func, retries=5, delay=0.1):
        """Retry file I/O operations to handle Windows file locking/permission errors."""
        last_exception = None
        for i in range(retries):
            try:
                return func()
            except (PermissionError, OSError) as e:
                last_exception = e
                if i < retries - 1:
                    time.sleep(delay + random.random() * 0.1)  # Add jitter
        if last_exception:
            raise last_exception

    def _load(self):
        if self.status_file.exists():
            def load_op():
                with self.status_file.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            
            try:
                return self._retry_io(load_op)
            except Exception:
                pass
        return {
            "dataset": self.dataset_name,
            "user_id": self.user_id,
            "history": [],
        }

    def _write(self, payload):
        payload["dataset"] = self.dataset_name
        payload["user_id"] = self.user_id
        
        def write_op():
            with self.status_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        
        try:
            self._retry_io(write_op)
        except Exception as e:
            print(f"Failed to write status file after retries: {e}")

    def _event(self, phase: str, message: str, state: str, completed: bool = False):
        return {
            "phase": phase,
            "message": message,
            "state": state,
            "completed": completed,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def update(self, phase: str, message: str, completed: bool = False):
        data = self._load()
        event = self._event(
            phase=phase,
            message=message,
            state="completed" if completed else "running",
            completed=completed,
        )
        data.setdefault("history", []).append(event)
        data["current"] = event
        self._write(data)

    def complete(self, message: str = "Training finished"):
        self.update("finished", message=message, completed=True)

    def error(self, message: str):
        data = self._load()
        event = self._event(phase="error", message=message, state="error", completed=False)
        data.setdefault("history", []).append(event)
        data["current"] = event
        self._write(data)


"""Quick test for WAL."""
import tempfile
from pathlib import Path
from clade.wal.logger import WALManager

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / 'wal'
    wm = WALManager(path, group_commit_enabled=False)
    print(f'Created WAL at {path}')
    print(f'Current LSN: {wm.current_lsn}')

    lsn = wm.log_begin(1)
    print(f'Logged begin, LSN: {lsn}')

    wm.flush()
    print('Flushed')

    wm.close()
    print('Closed successfully')

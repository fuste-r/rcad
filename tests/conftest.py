import os
import tempfile

# Keep tests from reading or writing the real user settings and recovery copies.
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="rcad-test-config-")
os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp(prefix="rcad-test-data-")  # recovery copies
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

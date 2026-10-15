import os
import tempfile

# Keep tests from reading or writing the real user settings.
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="rcad-test-config-")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

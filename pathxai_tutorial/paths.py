from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "vendor" / "report_engine"

def activate_engine():
    """Import only the report engine shipped in this tutorial checkout."""
    expected = (ENGINE / "pathxai_demo" / "__init__.py").resolve()
    existing = sys.modules.get("pathxai_demo")
    if existing is not None and Path(existing.__file__).resolve() != expected:
        raise RuntimeError("Another pathxai_demo package is already imported. Restart this kernel.")
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))


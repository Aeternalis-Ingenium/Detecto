from __future__ import annotations

# Let users know if they're missing any of our hard dependencies
__hard_dependencies = ("matplotlib", "numpy", "pandas", "scipy")
__missing_dependencies = []

for __dependency in __hard_dependencies:
    try:
        __import__(__dependency)
    except ImportError as _e:  # pragma: no cover
        __missing_dependencies.append(f"{__dependency}: {_e}")

if __missing_dependencies:  # pragma: no cover
    raise ImportError("Unable to import required dependencies:\n" + "\n".join(__missing_dependencies))
del __hard_dependencies, __dependency, __missing_dependencies


from src.detecto.models.detectors.factory import init_detecto
from src.detecto.models.notifications.factory import get_notification
from src.detecto.standalone.pot_detecto import (
    compute_extreme_anomaly_threshold,
    compute_pot_threshold,
    detect_extreme_anomaly,
    extract_pot_data,
    fit_pot_data,
    set_gpd_params,
)

__version__ = "0.1.0"

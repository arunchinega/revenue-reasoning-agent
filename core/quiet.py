"""
core/quiet.py — silence known-harmless library chatter (TF/Keras deprecations,
oneDNN banner, cmdstanpy INFO, sklearn/statsmodels warnings) so consoles and
demos stay clean. Import FIRST, before anything that pulls in TensorFlow.

Nothing here changes numerical behavior: oneDNN stays ON (we only mute the
banner text, not the optimization); warnings are hidden, not fixed — real
errors still raise normally.
"""
from __future__ import annotations

import logging
import os
import warnings


def quiet() -> None:
    # TensorFlow C++ log level: 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR spam
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # absl (the I0000 ... port.cc oneDNN banner)
    try:
        import absl.logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:  # noqa: BLE001
        pass
    # keras/tf python-side deprecation warnings (tf.reset_default_graph etc.)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*reset_default_graph.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            module=r"(keras|tensorflow).*")
    # cmdstanpy chain INFO chatter (Prophet)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    # generic statistical-library noise
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning,
                            module=r"(statsmodels|sklearn|prophet).*")


quiet()

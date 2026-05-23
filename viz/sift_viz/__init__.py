from __future__ import annotations

import logging
import sys

from sift_viz.app import SiftViz


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[sift-viz] %(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )
    app = SiftViz()
    app.mainloop()

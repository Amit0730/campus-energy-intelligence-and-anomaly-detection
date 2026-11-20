"""Rebuild the published day-ahead experiment and assessment downloads."""

from src.day_ahead import build
from src.assessment import export

if __name__ == "__main__":
    build()
    export()

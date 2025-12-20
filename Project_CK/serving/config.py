from __future__ import annotations

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from fastapi.templating import Jinja2Templates

GOC_DU_AN = Path(__file__).resolve().parent
THU_MUC_TEMPLATE = GOC_DU_AN / "templates"

templates = Jinja2Templates(directory=str(THU_MUC_TEMPLATE))

from dl_model.db.coco import Chart
from dl_model.db.coco import MSCOCO
from dl_model.db.coco import SKU
from dl_model.db.coco import Pie, Line, Bar, Cls, LineCls, LineClsReal
datasets = {
    "Chart": Chart,
    "MSCOCO": MSCOCO,
    "SKU110": SKU,
    "Pie": Pie,
    "Line": Line,
    "Bar": Bar,
    "Cls": Cls,
    "LineCls": LineCls,
    "LineClsReal": LineClsReal
}


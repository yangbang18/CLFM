from .CLLoss import CLLoss, oEWC
from typing import Union, Optional
from argparse import Namespace
from zeronlg import LossManager


def get_loss_class(
        CLL: bool = True,
        class_name: Optional[str] = None, 
        args: Optional[Namespace] = None
        ) -> Union[CLLoss, oEWC, LossManager]:
    if not CLL:
        return LossManager
    if class_name:
        assert class_name in ['CLLoss', 'oEWC']
        return eval(class_name)
    if args and getattr(args, 'fisher_penalty_scale', 0.0) > 0:
        return oEWC
    return CLLoss

from .adversarial_margin import AdversarialMarginLoss, classification_margin
from .mdb import MultiDomainBalancedLoss

__all__ = [
    "AdversarialMarginLoss",
    "MultiDomainBalancedLoss",
    "classification_margin",
]

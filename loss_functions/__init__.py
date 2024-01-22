
from .abstention_loss import DacLossPid
from .active_passive_loss import (
    NCEandRCE, GeneralizedCrossEntropy, NLNL, NFLandRCE, FocalLoss, CrossEntropy
)
from .misc import f_score
from torch.nn import L1Loss
from .elr import ELR, ELRPlus
from .polyloss import PolyLoss
from .simclr import info_nce_loss
from .isomax import IsoMaxLossSecondPart
from .asymetric_loss import NCEandAGCE, AGCELoss
from .isomaxplus import IsoMaxPlusLossSecondPart
from .consistency import kl_div, CrossEntropy as C_CrossEntropy
from .evidential_losses import EdLLogLoss, EdLMSELoss, EdLDigammaLoss
from .sup_con_loss import SupConLoss
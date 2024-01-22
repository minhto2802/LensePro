from utils.get_loss_function import get_loss_function
from training_strategy.pseudo_label import PseudoLabel
from training_strategy.consistency import ConsistencyRegularization
# from .ict import ICT
# from .vat import VAT


def gen_ssl_alg(cfg):
    name = cfg.name

    if name == "ict":  # mixed target <-> mixed input
        # return ICT(
        #     cfg.consistency,
        #     cfg.threshold,
        #     cfg.sharpen,
        #     cfg.temp_softmax,
        #     cfg.alpha
        # )
        raise NotImplementedError

    elif name == "cr":  # base augment <-> another augment
        return ConsistencyRegularization(
            get_loss_function(cfg.consistency),
            cfg.threshold,
            cfg.sharpen,
            cfg.temp_softmax
        )
    elif name == "pl":  # hard label <-> strong augment
        return PseudoLabel(
            get_loss_function(cfg.consistency),
            cfg.threshold,
            cfg.sharpen,
            cfg.temp_softmax
        )
    elif name == "vat":  # base augment <-> adversarial
        # from ..consistency import builder
        # return VAT(
        #     cfg.consistency,
        #     cfg.threshold,
        #     cfg.sharpen,
        #     cfg.temp_softmax,
        #     builder.gen_consistency(cfg.consistency, cfg),
        #     cfg.eps,
        #     cfg.xi,
        #     cfg.vat_iter
        # )
        raise NotImplementedError
    else:
        raise NotImplementedError

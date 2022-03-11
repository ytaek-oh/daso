import lib.algorithm as algorithm
from lib.config import get_cfg
from lib.engine import default_argument_parser, default_setup


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    cfg.freeze()

    # SSL algorithm
    trainer = algorithm.__dict__[cfg.ALGORITHM.NAME](cfg)
    if cfg.RESUME:
        trainer.load_checkpoint(cfg.RESUME)

    if args.eval_only:
        val_top1, test_top1 = trainer.evaluate()
        return

    trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)

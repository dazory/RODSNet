from __future__ import absolute_import, division, print_function

from options import Options

options = Options()
opts = options.parse()
from trainer import *
from utils.wandb_logger import WandbLogger


def parse_wandb_init_kwargs(opt):
    wandb_init_kwargs = dict(project="RODSNet", entity="kaist-url-ai28",
                             name=f"{opt.dataset}-{opt.model}-e{opt.epochs}",
                             config=dict(dataset=f"{opt.dataset}",
                                         model=f"{opt.model}",
                                         epochs=f"{opt.epochs}"))
    return wandb_init_kwargs

if __name__ == '__main__':

    trainer = Trainer(opts)

    wandb_logger = WandbLogger(init_kwargs=parse_wandb_init_kwargs(opts),
                               use_wandb=opts.wandb)
    trainer.set_wandb_logger(logger=wandb_logger)

    if opts.test_only:
        if opts.resume is None:
            raise RuntimeError("=> no checkpoint found...")
        else:
            print("checkpoint found at '{}' \n" .format(opts.resume))
        trainer.test()
    else:
        trainer.wandb_logger.before_run()
        for epoch in range(trainer.opts.start_epoch, trainer.opts.epochs):
            trainer.wandb_logger.before_train_epoch()
            trainer.train()
            trainer.wandb_logger.before_val_epoch()
            trainer.validate()
            trainer.scheduler.step()
            trainer.cur_epochs += 1
        trainer.wandb_logger.after_run()

        print('=> End training\n\n')
        trainer.writer.close()
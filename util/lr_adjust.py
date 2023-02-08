import math


class Adjust_learning_rate(object):
    def __init__(self, config):
        self.config = config

    def adjust_lr(self, optimizer=None, iter_lr=None, total_iters=None, epoch=None):
        if self.config.lr_policy == 'poly':
            current_lr = self.lr_poly(iter_lr, total_iters)
        elif self.config.lr_policy == 'lambda':
            current_lr = self.lr_lambda(epoch)
        elif self.config.lr_policy == 'cosine':
            current_lr = self.lr_cosine(iter_lr, total_iters)

        optimizer.param_groups[0]['lr'] = current_lr

        return current_lr

    def lr_poly(self, i_iter, total_iters):
        lr = self.config.learning_rate * ((1 - float(i_iter) / total_iters) ** self.config.power)
        return lr

    def lr_lambda(self, epoch):
        lr = self.config.learning_rate * self.config.lr_gamma ** (epoch // self.config.lr_decay_epochs)
        return lr

    def lr_cosine(self, step, total_iters):
        start_lr = 0.00001
        max_lr = self.config.learning_rate
        min_lr = 1e-6
        warm_steps = self.config.warm_steps
        total_steps = total_iters

        if step < warm_steps:
            lr = ((max_lr - start_lr) * step) / warm_steps + start_lr
        else:
            lr = max_lr * (math.cos(math.pi * (step - warm_steps) / (total_steps - warm_steps)) + 1) / 2

        lr = max(lr, min_lr)

        return lr

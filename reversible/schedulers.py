import numpy as np


class ScheduledOptimizer(object):
    def __init__(self, scheduler, optimizer,
                 schedule_weight_decay):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.schedule_weight_decay = schedule_weight_decay
        self.initial_lrs = list(map(
            lambda group: group['lr'], optimizer.param_groups))
        self.initial_weight_decays = list(map(
            lambda group: group['weight_decay'], optimizer.param_groups))
        self.i_update = 0

    def step(self):
        for group, initial_lr, initial_wd in zip(
                self.optimizer.param_groups,
                self.initial_lrs,
                self.initial_weight_decays):
            group['lr'] = self.scheduler.get_lr(initial_lr, self.i_update)
            if self.schedule_weight_decay:
                group['weight_decay'] = self.scheduler.get_weight_decay(
                    initial_wd, self.i_update)
        self.optimizer.step()
        self.i_update += 1

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()


class CosineAnnealing(object):
    def __init__(self, n_updates_per_period,):
        if not hasattr(n_updates_per_period, '__len__'):
            n_updates_per_period = [n_updates_per_period]
        assert np.all(np.array(n_updates_per_period) > 0)
        self.update_period_boundaries = np.cumsum(n_updates_per_period)
        self.update_period_boundaries = np.concatenate((
            [0], self.update_period_boundaries))

    def get_lr(self, initial_val, i_update):
        assert i_update < self.update_period_boundaries[-1], (
            "More updates ({:d}) than expected ({:d})".format(
                i_update, self.update_period_boundaries[-1] - 1))
        i_end_period = np.searchsorted(self.update_period_boundaries,
                                       i_update, side='right')
        assert i_end_period > 0
        i_start_update = self.update_period_boundaries[i_end_period - 1]
        i_end_update = self.update_period_boundaries[i_end_period]
        i_update = i_update - i_start_update
        assert i_update >= 0
        n_updates_this_period = i_end_update - i_start_update
        fraction_period = i_update / np.float64(n_updates_this_period)
        return initial_val * (0.5 * np.cos(np.pi * fraction_period) + 0.5)

    def get_weight_decay(self, initial_val, i_update):
        return self.get_lr(initial_val, i_update)


def cut_cos(x):
    """At the moment accepts value ebtween 0 and 1 not times pi like cosine!"""
    assert np.all(x >= 0)
    assert np.all(x <= 1)
    lower_part = np.cos(2*x*np.pi*(30/160.0))
    higher_part = np.cos(2*(x-0.5)* np.pi* (30/160.0) + 130/160.0*np.pi)
    result = (lower_part * (x < 0.5) + (higher_part * (x >= 0.5)))
    return result


class CutCosineAnnealing(object):
    def __init__(self, n_updates_per_period,):
        if not hasattr(n_updates_per_period, '__len__'):
            n_updates_per_period = [n_updates_per_period]
        assert np.all(np.array(n_updates_per_period) > 0)
        self.update_period_boundaries = np.cumsum(n_updates_per_period)
        self.update_period_boundaries = np.concatenate((
            [0], self.update_period_boundaries))

    def get_lr(self, initial_val, i_update):
        assert i_update < self.update_period_boundaries[-1], (
            "More updates ({:d}) than expected ({:d})".format(
                i_update, self.update_period_boundaries[-1] - 1))
        i_end_period = np.searchsorted(self.update_period_boundaries,
                                       i_update, side='right')
        assert i_end_period > 0
        i_start_update = self.update_period_boundaries[i_end_period - 1]
        i_end_update = self.update_period_boundaries[i_end_period]
        i_update = i_update - i_start_update
        assert i_update >= 0
        n_updates_this_period = i_end_update - i_start_update
        fraction_period = i_update / np.float64(n_updates_this_period)
        return initial_val * (0.5 * cut_cos(fraction_period) + 0.5)

    def get_weight_decay(self, initial_val, i_update):
        return self.get_lr(initial_val, i_update)


class DivideSqrtUpdates(object):
    def get_lr(self, initial_val, i_update):
        return initial_val / np.sqrt(i_update + 1)

    def get_weight_decay(self, initial_val, i_update):
        return self.get_lr(initial_val, i_update)


class DivideUpdates(object):
    def get_lr(self, initial_val, i_update):
        return initial_val / (i_update + 1)

    def get_weight_decay(self, initial_val, i_update):
        return self.get_lr(initial_val, i_update)
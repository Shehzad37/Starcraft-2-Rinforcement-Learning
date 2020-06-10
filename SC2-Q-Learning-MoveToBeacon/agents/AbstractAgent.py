from abc import abstractmethod
from pysc2.lib import actions, features
import numpy as np


class AbstractAgent:

    """Sc2 Actions"""
    _MOVE_SCREEN = actions.FUNCTIONS.Move_screen
    _NO_OP = actions.FUNCTIONS.no_op()
    _SELECT_ARMY = actions.FUNCTIONS.select_army("select")

    def __init__(self, screen_size):
        self.screen_size = screen_size

    @abstractmethod
    def step(self, obs): ...

    @abstractmethod
    def save_model(self, filename): ...

    @abstractmethod
    def load_model(self, filename): ...

    def get_beacon(self, obs):
        """Returns the unit obj representation of the beacon"""
        beacon = next(unit for unit in obs.observation.feature_units
                      if unit.alliance == features.PlayerRelative.NEUTRAL)
        return beacon

    def get_marine(self, obs):
        """Returns the unit obj representation of the marine"""
        marine = next(unit for unit in obs.observation.feature_units
                      if unit.alliance == features.PlayerRelative.SELF)
        return marine

    def get_unit_pos(self, unit):
        """Returns the (x, y) position of a unit obj"""
        return np.array([unit.x, unit.y])

    def _xy_offset(self, start, offset_x, offset_y):
        """Return point (x', y') offset from point start.
        Pays attention to not set the point off beyond the screen border"""
        dest = start + np.array([offset_x, offset_y])

        if dest[0] < 0:
            dest[0] = 0
        elif dest[0] >= self.screen_size:
            dest[0] = self.screen_size - 1

        if dest[1] < 0:
            dest[1] = 0
        elif dest[1] >= self.screen_size:
            dest[1] = self.screen_size - 1

        return dest

from pysc2.lib import actions, features, units
import numpy as np
import math


def get_marine(obs):
    marine = next(unit for unit in obs.observation.feature_units
                  if unit.alliance == features.PlayerRelative.SELF)

    if not marine.unit_type == units.Terran.Marine:
        pass

    return marine


def get_beacon(obs):

    beacon = next(unit for unit in obs.observation.feature_units
                  if unit.alliance == features.PlayerRelative.NEUTRAL)

    return beacon


def get_units(obs, type):
    selected_units = [unit for unit in obs.observation.feature_units if unit.unit_type == type]
    return selected_units


def move_screen(x, y):
    return actions.FUNCTIONS.Move_screen("now", (x, y))


def select_other_marine(obs):
    m = next(unit for unit in get_units(obs, units.Terran.Marine) if not unit.is_selected)
    return select_point(m)


def select_point(pos):
    return actions.FUNCTIONS.select_point("select", (pos.x, pos.y))


def get_selected_marine(obs):
    m = next(unit for unit in get_units(obs, units.Terran.Marine) if unit.is_selected)
    return m


def get_neutral_units(obs):
    neutral_units = [unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.NEUTRAL]
    return neutral_units


def preprocess_channels(obs):

    channels = obs.observation.feature_screen
    state_size = channels.shape

    data = np.ndarray(shape=(state_size[0], state_size[1], state_size[2]))

    c = s1 = s2 = 0

    while c < state_size[0]:
        while s1 < state_size[1]:
            while s2 < state_size[2]:
                data[c, s1, s2] = channels[c, s1, s2]
                s2 += 1
            s1 += 1
        c += 1

    return data


def state_of_marine(marine, beacon, screen, distance_window):

    dist_x = beacon.x - marine.x
    dist_y = beacon.y - marine.y

    return discretize_distance(dist_x, screen, distance_window), discretize_distance(dist_y, screen, distance_window, 0.8)


def discretize_distance(dist, screen, distance_window, factor=1.0):

    if distance_window == -1:
        return discretize_distance_float(dist, screen, factor)

    percentual_val = round(dist / screen, 2)
    disc_val = math.ceil(percentual_val / distance_window * 100)
    return disc_val

    # return math.ceil(round(dist / self.max_distance, 2) / self._DISTANCE_WINDOW * 100)
    #
    # tmp_val = dist / self._DISTANCE_WINDOW
    # return_val = math.ceil(tmp_val)
    # return return_val


def discretize_distance_float(dist, screen, factor=1.0):

    disc_dist = dist/(screen * factor)
    return disc_dist


from sls.agents.AbstractAgent import AbstractAgent


class BasicAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(BasicAgent, self).__init__(screen_size)

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

    def step(self, obs):

        if obs.first():
            return self._SELECT_ARMY

        marine = self.get_unit_pos(self.get_marine(obs))
        beacon = self.get_unit_pos(self.get_beacon(obs))
        distance = beacon - marine

        destination = [0, 0]

        if distance[0] > 0:
            destination[0] = self.screen_size
        elif distance[0] < 0:
            destination[0] = -self.screen_size

        if distance[1] > 0:
            destination[1] = self.screen_size
        elif distance[1] < 0:
            destination[1] = -self.screen_size

        return self._MOVE_SCREEN("now", self._xy_offset(marine, destination[0], destination[1]))


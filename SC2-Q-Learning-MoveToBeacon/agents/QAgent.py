from sls.learning.Q_Learning import Q_Learning
from sls.agents import AbstractAgent
from sls.minigames.utils import state_of_marine


class QAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QAgent, self).__init__(screen_size)

        self.learner = Q_Learning(range(8),
                                  epsilon=1,
                                  descending_epsilon=True,
                                  descend_epsilon_until=500,
                                  alpha=0.2)
        self.eps = 1
        self.bool = True

    def step(self, obs):

        if obs.first():
            self.learner.descend_epsilon()
            return self._SELECT_ARMY

        beacon = self.get_beacon(obs)
        marine = self.get_marine(obs)
        state = state_of_marine(marine, beacon, self.screen_size, 10)

        action = self.learner.get_action(state, obs.reward)
        if self.eps % 1920 == 0:
            self.learner.print_q()
        if self.bool and self.eps % 240 == 0:
            self.learner.print_q()
            self.bool = False
        dest = []

        if action == 0:
            dest = [0, self.screen_size]
        elif action == 1:
            dest = [self.screen_size, 0]
        elif action == 2:
            dest = [0, -self.screen_size]
        elif action == 3:
            dest = [-self.screen_size, 0]
        elif action == 4:
            dest = [self.screen_size, self.screen_size]
        elif action == 5:
            dest = [self.screen_size, -self.screen_size]
        elif action == 6:
            dest = [-self.screen_size, self.screen_size]
        elif action == 7:
            dest = [-self.screen_size, -self.screen_size]
        self.eps += 1
        return self._MOVE_SCREEN("now", self._xy_offset(self.get_unit_pos(marine), dest[0], dest[1]))

    def save_model(self, filename):
        self.learner.save_q_table(filename + '/model.pkl')

    def load_model(self, filename):
        self.learner.load_model(filename + '/model.pkl')

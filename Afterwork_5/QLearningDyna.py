import time
import mdptoolbox.example
import numpy as np
import matplotlib.pyplot as plt


class QLearner(object):

    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.q_table = np.zeros((num_states, num_actions))
        self.R = np.zeros((num_states, num_actions))
        self.T = np.zeros((num_states, num_actions, num_states))
        self.T_count = np.full((num_states, num_actions, num_states), 1e-10)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        # Decide whether to take a random action or not
        if np.random.rand() <= self.rar:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(self.q_table[s])

        # Update random action rate
        self.rar *= self.radr

        # Save s and a
        self.s = s
        self.a = action

        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """
        self.q_table[self.s, self.a] = (1 - self.alpha) * self.q_table[self.s, self.a] + self.alpha * (
                r + self.gamma * np.max(self.q_table[s_prime]))

        # Update T_count table
        self.T_count[self.s, self.a, s_prime] += 1

        # Dyna part
        self.T[self.s, self.a, s_prime] = self.T_count[self.s, self.a, s_prime] / np.sum(self.T_count[self.s, self.a])
        self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

        hallucinated_s = np.random.randint(0, self.num_states, size=self.dyna)
        hallucinated_a = np.random.randint(0, self.num_actions, size=self.dyna)

        for i in range(self.dyna):
            hallucinated_s_i = hallucinated_s[i]
            hallucinated_a_i = hallucinated_a[i]
            hallucinated_s_prime = np.argmax(self.T[hallucinated_s_i, hallucinated_a_i])
            hallucinated_r = self.R[hallucinated_s_i, hallucinated_a_i]
            self.q_table[hallucinated_s_i, hallucinated_a_i] = (1 - self.alpha) * self.q_table[
                hallucinated_s_i, hallucinated_a_i] + self.alpha * (hallucinated_r + self.gamma * np.max(
                self.q_table[hallucinated_s_prime]))

        return self.querysetstate(s_prime)


class TestQLearnerMap(object):

    # print out the map
    def printmap(self, data):
        print("---------------------")
        for row in range(0, data.shape[0]):
            print("|", end='')
            for col in range(0, data.shape[1]):
                if data[row, col] == 0:  # Empty space
                    print(" |", end='')
                if data[row, col] == 1:  # Obstacle
                    print("O|", end='')
                if data[row, col] == 2:  # Start
                    print("*|", end='')
                if data[row, col] == 3:  # Goal
                    print("X|", end='')
                if data[row, col] == 4:  # Trail
                    print(".|", end='')
                if data[row, col] == 5:  # Quick sand
                    print("~|", end='')
                if data[row, col] == 6:  # Stepped in quicksand
                    print("@|", end='')
            print()
        print("---------------------")

    # find where the robot is in the map
    def getrobotpos(self, data):
        R = -999
        C = -999
        for row in range(0, data.shape[0]):
            for col in range(0, data.shape[1]):
                if data[row, col] == 2:
                    C = col
                    R = row
        if (R + C) < 0:
            print("warning: start location not defined")
        return R, C

    # find where the goal is in the map
    def getgoalpos(self, data):
        R = -999
        C = -999
        for row in range(0, data.shape[0]):
            for col in range(0, data.shape[1]):
                if data[row, col] == 3:
                    C = col
                    R = row
        if (R + C) < 0:
            print("warning: goal location not defined")
        return (R, C)

    # move the robot and report reward
    def movebot(self, data, oldpos, a):
        testr, testc = oldpos

        randomrate = 0.20  # how often do we move randomly
        quicksandreward = -100  # penalty for stepping on quicksand

        go_random = np.random.uniform(0.0, 1.0) <= randomrate

        # update the test location
        if a == 0:  # north
            if go_random:
                if np.random.uniform(0.0, 1.0) <= randomrate:
                    testc -= 1
                else:
                    testc += 1
            else:
                testr = testr - 1
        elif a == 1:  # east
            if go_random:
                if np.random.uniform(0.0, 1.0) <= randomrate:
                    testr -= 1
                else:
                    testr += 1
            else:
                testc = testc + 1
        elif a == 2:  # south
            if go_random:
                if np.random.uniform(0.0, 1.0) <= randomrate:
                    testc -= 1
                else:
                    testc += 1
            else:
                testr = testr + 1
        elif a == 3:  # west
            if go_random:
                if np.random.uniform(0.0, 1.0) <= randomrate:
                    testr -= 1
                else:
                    testr += 1
            else:
                testc = testc - 1

        reward = -1  # default reward is negative one
        # see if it is legal. if not, revert
        if testr < 0:  # off the map
            testr, testc = oldpos
        elif testr >= data.shape[0]:  # off the map
            testr, testc = oldpos
        elif testc < 0:  # off the map
            testr, testc = oldpos
        elif testc >= data.shape[1]:  # off the map
            testr, testc = oldpos
        elif data[testr, testc] == 1:  # it is an obstacle
            testr, testc = oldpos
        elif data[testr, testc] == 5:  # it is quicksand
            reward = quicksandreward
            data[testr, testc] = 6  # mark the event
        elif data[testr, testc] == 6:  # it is still quicksand
            reward = quicksandreward
            data[testr, testc] = 6  # mark the event
        elif data[testr, testc] == 3:  # it is the goal
            reward = 1  # for reaching the goal

        return (testr, testc), reward  # return the new, legal location

    # convert the location to a single integer
    def discretize(self, pos):
        return pos[0] * 10 + pos[1]

    def test(self, map, epochs, learner, verbose):
        # each epoch involves one trip to the goal
        startpos = self.getrobotpos(map)  # find where the robot starts
        goalpos = self.getgoalpos(map)  # find where the goal is
        scores = np.zeros((epochs, 1))
        for epoch in range(1, epochs + 1):
            total_reward = 0
            data = map.copy()
            robopos = startpos
            state = self.discretize(robopos)  # convert the location to a state
            action = learner.querysetstate(state)  # set the state and get first action
            count = 0
            while (robopos != goalpos) & (count < 100000):

                # move to new location according to action and then get a new action
                newpos, stepreward = self.movebot(data, robopos, action)
                if newpos == goalpos:
                    r = 1  # reward for reaching the goal
                else:
                    r = stepreward  # negative reward for not being at the goal
                state = self.discretize(newpos)
                action = learner.query(state, r)

                if data[robopos] != 6:
                    data[robopos] = 4  # mark where we've been for map printing
                if data[newpos] != 6:
                    data[newpos] = 2  # move to new location
                robopos = newpos  # update the location
                # if verbose: time.sleep(1)
                total_reward += stepreward
                count = count + 1
            if count == 100000:
                print("timeout")
            if verbose: self.printmap(data)
            if verbose: print(epoch, total_reward)
            scores[epoch - 1, 0] = total_reward
        return np.median(scores)

    # run the code to test a learner
    def test_code(self, filename='world01.csv'):

        verbose = True  # print lots of debug stuff if True

        inf = open(filename)
        data = np.array([s.strip().split(',') for s in inf.readlines()]).astype(np.int)
        originalmap = data.copy()  # make a copy so we can revert to the original map later

        if verbose:
            self.printmap(data)

        np.random.seed(5)

        learner = QLearner(num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.98,
                           radr=0.999, dyna=200)  # initialize the learner
        epochs = 500
        total_reward = self.test(data, epochs, learner, verbose)
        print(epochs, "median total_reward", total_reward)
        print(np.argmax(learner.q_table, axis=1))
        print_policy(world=world2mat(filename), policy=np.argmax(learner.q_table, axis=1))
        score = total_reward

        print("results for " + filename)
        print("score: " + str(score))


class TestQLearnerForest(object):

    def __init__(self, p, nbstates=5, steps=20, r0=4, r1=2, epochs=5):
        self.p = p
        self.nbstates = nbstates
        self.steps = steps
        self.r0 = r0
        self.r1 = r1
        self.epochs = epochs

    def move(self, nbstates, oldstate, a):
        newstate = oldstate
        reward = 0

        if np.random.uniform(0.0, 1.0) <= self.p:
            newstate = 0
            reward = 0

        else:
            # update the test location
            if a == 0:  # wait
                if newstate < nbstates - 1:
                    newstate += 1
                if newstate == nbstates - 1:
                    reward = self.r0
                else:
                    reward = 0

            elif a == 1:  # cut
                if newstate == nbstates - 1:
                    reward = self.r1
                elif newstate == 0:
                    reward = 0
                else:
                    reward = -1
                newstate = 0
            else:
                print("Unknown action")
                exit()

        return newstate, reward

    def test(self, nbstates, epochs, steps, learner, verbose):
        # each epoch involves one trip to the goal
        scores = np.zeros((epochs, 1))
        for epoch in range(1, epochs + 1):
            actions = np.zeros(steps)
            states = np.zeros(steps)
            total_reward = 0
            state = 0
            action = learner.querysetstate(state)  # set the state and get first action
            count = 0
            while count < steps:
                actions[count] = action
                states[count] = state
                state, r = self.move(nbstates, state, action)
                action = learner.query(state, r)
                total_reward += r
                count = count + 1
            if count == steps:
                print("timeout")
            if verbose: print(epoch, total_reward)
            if verbose: print("states: " + str(states))
            if verbose: print("action: " + str(actions))
            scores[epoch - 1, 0] = total_reward
        return np.median(scores)

    # run the code to test a learner
    def test_code(self):

        verbose = True  # print lots of debug stuff if True

        np.random.seed(5)

        learner = QLearner(num_states=self.nbstates, num_actions=2, alpha=0.2, gamma=0.9, rar=0.98,
                           radr=0.999, dyna=15)  # initialize the learner
        total_reward = self.test(self.nbstates, self.epochs, self.steps, learner, verbose)
        print(self.epochs, "median total_reward", total_reward)
        print(np.argmax(learner.q_table, axis=1))
        score = total_reward

        print("score: " + str(score))


def world2mat(filename='world01.csv'):
    inf = open(filename)
    return np.array([s.strip().split(',') for s in inf.readlines()]).astype(np.int)


def get_start(world):
    i, j = np.where(world == 2)
    i = i[0]
    j = j[0]
    return i, j


def get_goal(world):
    i, j = np.where(world == 3)
    i = i[0]
    j = j[0]
    return i, j


def print_map(world):
    print("---------------------")
    for row in range(0, world.shape[0]):
        print("|", end='')
        for col in range(0, world.shape[1]):
            if world[row, col] == 0:  # Empty space
                print(" |", end='')
            if world[row, col] == 1:  # Obstacle
                print("O|", end='')
            if world[row, col] == 2:  # Start
                print("*|", end='')
            if world[row, col] == 3:  # Goal
                print("X|", end='')
            if world[row, col] == 5:  # Quick sand
                print("~|", end='')
            if world[row, col] == 6:  # Stepped in quick sand
                print("@|", end='')
            if world[row, col] == 10:  # Path
                print(".|", end='')
            if world[row, col] == 20:  # Arrow north
                print("\u2191|", end='')
            if world[row, col] == 21:  # Arrow east
                print("\u2192|", end='')
            if world[row, col] == 22:  # Arrow south
                print("\u2193|", end='')
            if world[row, col] == 23:  # Arrow west
                print("\u2190|", end='')
        print()
    print("---------------------")


def print_policy(world, policy):
    arrows = world.copy()
    k = 0
    for i in range(arrows.shape[0]):
        for j in range(arrows.shape[1]):
            if arrows[i, j] != 1:
                arrows[i, j] = 20 + policy[k]
            k += 1
    print_map(arrows)


def print_policy_differences(world, vi_policy, pi_policy):
    arrows = world.copy()
    k = 0
    for i in range(arrows.shape[0]):
        for j in range(arrows.shape[1]):
            if arrows[i, j] != 1:
                if pi_policy[k] != vi_policy[k]:
                    arrows[i, j] = 20 + pi_policy[k]
                else:
                    arrows[i, j] = 0
            k += 1
    print_map(arrows)


def get_word_coord_from_state(world, s):
    return s // world.shape[0], s % world.shape[1]


def get_world_value_from_state(world, s):
    return world[s // world.shape[0], s % world.shape[1]]


def get_state_from_world_coord(world, i, j):
    return i * world.shape[0] + j


def get_world_value_after_action(world, s, a):
    i, j = get_word_coord_from_state(world, s)
    prec_i, prec_j = i, j
    if a == 0:  # north
        i -= 1
    elif a == 1:  # east
        j += 1
    elif a == 2:  # south
        i += 1
    elif a == 3:  # west
        j -= 1
    if i < 0:
        i = 0
    if i == world.shape[0]:
        i = world.shape[0] - 1
    if j < 0:
        j = 0
    if j == world.shape[1]:
        j = world.shape[1] - 1
    if world[i, j] == 1:
        i, j = prec_i, prec_j
    return world[i, j]


def get_reward_from_world_value(world_value):
    if world_value == 0:
        return -1
    elif world_value == 1:
        return -1
    elif world_value == 2:
        return -1
    elif world_value == 3:
        return 1
    elif world_value == 5:
        return -100
    else:
        print("Error: unknown world value")
        exit()


def get_T_from_world(num_states, num_actions, world, p=0.2):
    """
    actions: 0 north, 1 east, 2 south, 3 west
    (1-p) chance to execute action, and p chance to move in right angle
    """
    T = np.zeros((num_actions, num_states, num_states))
    rows = world.shape[0]
    cols = world.shape[1]
    # start_i, start_j = np.where(world == 2)
    # start_i = start_i[0]
    # start_j = start_j[0]
    for s in range(0, num_states):
        i, j = get_word_coord_from_state(world, s)

        # If goal reached, stay on it
        if world[i, j] == 3:
            T[:, s, s] = 1
            continue

        # # If moved to quicksand, go back to start point
        # if world[i, j] == 5:
        #     T[:, s, start_state] = 1
        #     continue

        # move north
        if i - 1 < 0 or world[i - 1, j] == 1:
            T[0, s, s] += 1 - p  # action north
            T[1, s, s] += p / 2  # action east
            T[3, s, s] += p / 2  # action west
        else:
            sprime = get_state_from_world_coord(world, i - 1, j)
            T[0, s, sprime] += 1 - p  # action north
            T[1, s, sprime] += p / 2  # action east
            T[3, s, sprime] += p / 2  # action west

        # move east
        if j + 1 == cols or world[i, j + 1] == 1:
            T[1, s, s] += 1 - p  # action east
            T[0, s, s] += p / 2  # action north
            T[2, s, s] += p / 2  # action south
        else:
            sprime = get_state_from_world_coord(world, i, j + 1)
            T[1, s, sprime] += 1 - p  # action east
            T[0, s, sprime] += p / 2  # action north
            T[2, s, sprime] += p / 2  # action south

        # move south
        if i + 1 == rows or world[i + 1, j] == 1:
            T[2, s, s] += 1 - p  # action south
            T[1, s, s] += p / 2  # action east
            T[3, s, s] += p / 2  # action west
        else:
            sprime = get_state_from_world_coord(world, i + 1, j)
            T[2, s, sprime] += 1 - p  # action south
            T[1, s, sprime] += p / 2  # action east
            T[3, s, sprime] += p / 2  # action west

        # move west
        if j - 1 < 0 or world[i, j - 1] == 1:
            T[3, s, s] += 1 - p  # action west
            T[0, s, s] += p / 2  # action north
            T[2, s, s] += p / 2  # action south
        else:
            sprime = get_state_from_world_coord(world, i, j - 1)
            T[3, s, sprime] += 1 - p  # action west
            T[0, s, sprime] += p / 2  # action north
            T[2, s, sprime] += p / 2  # action south

    return T


def get_R_from_world(num_states, num_actions, world, T):
    """
    actions: 0 north, 1 east, 2 south, 3 west
    """
    R = np.zeros((num_states, num_actions))
    for s in range(0, num_states):
        for a in range(0, num_actions):
            # R[s, a] = get_reward_from_world_value(get_world_value_after_action(world, s, a))
            for sprime in range(0, num_states):
                R[s, a] += T[a, s, sprime] * get_reward_from_world_value(get_world_value_from_state(world, sprime))
    return R


if __name__ == '__main__':
    T, R = mdptoolbox.example.forest(S=10, r1=4, r2=-2, p=0.2)  # Wait=0, Cut=1
    vi = mdptoolbox.mdp.ValueIteration(transitions=T, reward=R, discount=0.9, max_iter=100)
    # vi.setVerbose()
    vi.run()
    print("MILLISECONDS VALUE IT: " + str(vi.time * 1000))
    print(vi.policy)
    pi = mdptoolbox.mdp.PolicyIteration(transitions=T, reward=R, discount=0.9, max_iter=100)
    # pi.setVerbose()
    pi.run()
    print("MILLISECONDS POLICY IT: " + str(pi.time * 1000))
    print(pi.policy)

    start = time.time()
    TestQLearnerForest(p=0.3, nbstates=10, steps=20, r0=4, r1=2, epochs=500).test_code()
    print("MILLISECONDS QLEARNING: " + str((time.time() - start) * 1000))

    filename = 'world01.csv'
    world = world2mat(filename)
    print_map(world)
    T = get_T_from_world(world.shape[0] * world.shape[1], 4, world, p=0.2)
    R = get_R_from_world(world.shape[0] * world.shape[1], 4, world, T)

    print(R[50, :])

    vi = mdptoolbox.mdp.ValueIteration(transitions=T, reward=R, discount=0.9, max_iter=500)
    vi.setVerbose()
    vi.run()
    print("MILLISECONDS VALUE IT: " + str(vi.time * 1000))
    print("NUMBER VALUE IT: " + str(vi.iter))
    print_policy(world, np.array(vi.policy))

    pi = mdptoolbox.mdp.PolicyIteration(transitions=T, reward=R, discount=0.9, max_iter=500)
    # pi.setVerbose()
    pi.run()
    print("MILLISECONDS POLICY IT: " + str(pi.time * 1000))
    print("NUMBER POLICY IT: " + str(pi.iter))
    print_policy(world, np.array(pi.policy))

    print(pi.policy == vi.policy)
    print_policy_differences(world, np.array(vi.policy), np.array(pi.policy))

    start = time.time()
    TestQLearnerMap().test_code(filename)
    print("MILLISECONDS QLEARNING: " + str((time.time() - start) * 1000))

"""
Below, is the updated code for the class Birds. It is mainly the same as that found in the reference repository (with more detailed comments), but wwith 2 additions:
1) The function "is_in_collision" that identifies if two birds are in close proximity/collision
2) The cost defined by "collision_penalty" inside the function "reward" to penalize collision events
"""

#importing the necessary packages
import numpy as np
from random import randint, choice, choices
from scipy.spatial import KDTree
from bisect import bisect_left
import time

#calling the function that performs
from q_learning import Qfunction

#Initializing the parameters related to birds
#There parameters can be varied and one can observe their effect by looking at the results obtained
d = 100  # Observation radius
M = 2    # Max neigbours observed by each bird
R = 5    # Maximum reward signal

#Initializing the Q-learning related parameters
#These also can be varied
alpha   = 0.1   #learning rate
gamma   = 0.9   #discount factor
epsilon = 0.5   #exploration probability

#Initializing the number of directions of observability for each bird
#This is obtained by discretizing the possible angles between [-pi, pi]
#Note: state space explodes for exponent > 3
NO_DIRS = 2 ** 3    #choosing 8 in total
DIRS = np.linspace(-np.pi, np.pi, NO_DIRS + 1)[:NO_DIRS]
DIRS_INDS = list(range(NO_DIRS))

#Calculates the x and y components of the unit step vector
STEPS = [np.array([round(np.cos(theta), 5), round(np.sin(theta), 5)]) for theta in DIRS]

TRESHOLD = 0.9 * np.linalg.norm(STEPS[0] + STEPS[(NO_DIRS//2) + 1])

#Set of actions:
#V: stands for Vicsek-like interaction where birds will align themselves along the average flight direction - this is typical of followers
#I: stands for Instinct where birds will follow their intinct - flying along one of the 4 cardinal directions (N, S, E, W) - this is typical of leaders
A = ['V', 'I']

#Set of observations ---> this coincides with the state space
O = range((M + 1) ** NO_DIRS)

#Initializing dictionary containing the 4 cardinal directions (N,S,E,W)
CARD_DIRS = {
    card_dir: i for card_dir, i in zip('WSEN', range(0, NO_DIRS, NO_DIRS//4))
}


def discrete_Vicsek(observation, strict=False):
    """
    Returns the index of DIRS that is closest to the average direction. If the
    sum of directions is zero (e.g. one north and one south), it will return
    NO_DIRS (= len(DIRS)) as an exception case.

    """
    v = np.array([0., 0.])
    for dir, n in observation.items():
        v += n * STEPS[dir]
    if np.linalg.norm(v) > TRESHOLD:
        theta = np.arctan2(v[1], v[0])
        i = bisect_left(DIRS, theta)
        if i == NO_DIRS:
            delta_1 = theta - DIRS[i - 1]
            delta_2 = np.pi - theta
            if abs(delta_1 - delta_2) < 1e-4:
                return NO_DIRS if strict else choice([i - 1, i])
            elif delta_1 < delta_2:
                return i - 1
            else:
                return 0
        elif i == 0:
            return 0
        else:
            delta_1 = theta - DIRS[i - 1]
            delta_2 = DIRS[i] - theta
            if abs(delta_1 - delta_2) < 1e-4:
                return NO_DIRS if strict else choice([i - 1, i])
            elif delta_1 < delta_2:
                return i - 1
            else:
                return i
    else:
        return NO_DIRS

def ternary(numbers):
    numbers = list(numbers)
    return sum(numbers[-1 * (i + 1)] * 3 ** i for i in range(len(numbers)))

def argmax(array):
    """ Returns the index of the maximum value of an array. If multiple maximum values
    exist, it returns one of them at random (unlike np.argmax).

    Inspired by https://stackoverflow.com/questions/17568612 """
    all_inds = [0]
    max_value = array[0]
    for i,value in enumerate(array[1:], start = 1):
        if value > max_value:
            all_inds = [i]
            max_value = value
        elif value == max_value:
            all_inds.append(i)

    if len(all_inds) == 1:
        return all_inds[0]
    else:
        return choice(all_inds)

"""
get_maj_obs and check_rotational_symmetry have been used to check whether
discrete_Vicsek works as expected. The first returns a list for each cardinal
direction with all the possible observations in which discrete_Vicsek points
in that direction (by symmetry, these lists should be equal in size). The second
checks the 90 degree rotational symmetry more thoroughly by iterating over all
possible observations, performing 90 degree rotations and checking whether
discrete_Vicsek rotates accordingly. Both make use of _to_obs and _rot.

As of 25/05/2020, discrete_Vicsek passes both tests.
"""

def _to_obs(i):
    """ Converts the observation index back to the observation dictionary. """
    if NO_DIRS == 4:
        return {n: int(a) for n,a in enumerate(f'{np.base_repr(i, M + 1):>04}')}
    elif NO_DIRS == 8:
        return {n: int(a) for n,a in enumerate(f'{np.base_repr(i, M + 1):>08}')}

def _rot(obs, n = 1):
    """ Returns obs n times rotated by 90 degrees. """
    if n == 0:
        return obs
    new_obs = {i: obs[(i - 2) % NO_DIRS] for i in range(NO_DIRS)}
    if n == 1:
        return new_obs
    elif n > 1 and type(n) == int:
        return _rot(new_obs, n - 1)

def get_maj_obs(print_sizes = True):
    maj = {card_dir: [] for card_dir in CARD_DIRS.keys()}
    for i in O:
        vic_dir = discrete_Vicsek(_to_obs(i), strict = True)
        for card_dir, i_dir in CARD_DIRS.items():
            if vic_dir == i_dir:
                maj[card_dir].append(i)

    if print_sizes:
        print([len(dirs) for dirs in maj.values()])

    return maj

def check_rotational_symmetry():
    if NO_DIRS != 8:
        raise ValueError('This function only works for NO_DIRS == 8')
    not_symmetric = []
    for i in O:
        obs = _to_obs(i)
        vic_dirs = [
            discrete_Vicsek(_rot(obs, n), strict = True) for n in range(4)
        ]
        if 8 in vic_dirs and vic_dirs != [8,8,8,8]:
            not_symmetric.append((obs, vic_dirs))
        elif 0 in vic_dirs:
            ind = vic_dirs.index(0)
            shifted = [vic_dirs[(i + ind) % 4] for i in range(4)]
            if not shifted == [0,2,4,6]:
                not_symmetric.append((obs, vic_dirs))
        elif 1 in vic_dirs:
            ind = vic_dirs.index(1)
            shifted = [vic_dirs[(i + ind) % 4] for i in range(4)]
            if not shifted == [1,3,5,7]:
                not_symmetric.append((obs, vic_dirs))
    if len(not_symmetric) == 0:
        print('discrete_Vicsek is completely rotationally symmetric')
    else:
        print('Asymmetry found, its indices are returned')
        return not_symmetric


class Birds(object):

    def __init__(self, numbirds, field_dims, action_space=A,
                 observation_space=O, leader_frac=0.25, reward_signal=R, gradient_reward=False, learning_alg='Q', alpha=alpha,
                 gamma=gamma, epsilon=epsilon, Q_file='', Q_tables=None, observation_radius=d, instincts=[], eps_decr=0, collision_distance = 1.0):

        # Initialize the birds and put them in the field
        #total nb of birds
        self.numbirds = numbirds

        #Assign the fraction of leaders
        self.leader_frac = leader_frac
        #Calculating the number of leaders
        self.leaders = int(self.numbirds * self.leader_frac)

        #Assigning the positions of the birds
        self.initialize_positions(field_dims)
        #Assigning the directions of the birds
        self.dirs = choices(DIRS_INDS, k=self.numbirds)

        #Assign collision distance
        self.collision_distance = collision_distance


        if instincts:
            if len(instincts) != self.numbirds:
                raise ValueError('Given list of instincts does not equal number of birds')
            self.instincts = instincts
        else:
            self.instincts = (
                self.leaders * ['E']   #leaders fly in the East direction by construction (for simplicity bc theta = 0)
                + choices(['N', 'S', 'W'], k=self.numbirds - self.leaders)   #followers get a  random direction
            )

        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_signal = reward_signal
        self.gradient = gradient_reward   #for gradient reward, but we will keep it False throughout the simulations
        self.observation_radius = observation_radius
        self.learning_alg = learning_alg  #Will be Q-learning

        self.policies = np.zeros([self.numbirds, len(O), len(self.action_space)])

        if self.learning_alg == 'pol_from_Q':
            if (not Q_file) and (type(Q_tables) != np.ndarray):
                raise Exception('No Q-values provided')
            else:
                if Q_file:
                    Q_tables = np.load(Q_file)
                for i in range(self.numbirds):
                    for s in range(self.policies.shape[1]):
                        self.policies[i, s, argmax(Q_tables[i, s])] = 1
            self.Q_tables = Q_tables
            self.Delta = self.calc_Delta()
        else:
            self.policies += 1 / len(self.action_space)

        self.observations = [dict() for _ in range(self.numbirds)]
        self.perform_observations()

        if self.learning_alg == 'Q':
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.eps_decr = eps_decr
            if self.eps_decr:
                self.eps_step = epsilon / eps_decr
            self.Qs = [Qfunction(alpha, gamma, self.observation_space, self.action_space) for _ in range(self.numbirds)]
        else:
            self.eps_decr = 0

    ### This is new: Function that identifies when two birds collide with one another
    def is_in_collision(self, i):   #can try for different values of the radius
        for j in range(self.numbirds):
            if j == i:
                continue
            dist = np.linalg.norm(self.positions[i] - self.positions[j])   #calculates the distance between two birds
            if dist < self.collision_distance:
                return True
        return False

    #Define a function that returns the parameters specific to the run
    def request_params(self):
        params = {
            'no_birds': self.numbirds,
            'action_space': self.action_space,
            'leader_frac': self.leader_frac,
            'observation_radius': self.observation_radius,
            'reward_signal': self.reward_signal,
            'learning_alg': self.learning_alg,
            'gradient_reward': self.gradient,
            'no_dirs': NO_DIRS
        }
        if self.learning_alg == 'Q':
            params['Q_params'] = {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        if self.learning_alg == 'pol_from_Q':
            params['Delta'] = self.Delta
        return params

    #Assigning birds' positions
    def initialize_positions(self, field_dims):
        self.positions = np.array([
            np.array([
                float(randint(*field_dims[0:2])),
                float(randint(*field_dims[2:4]))
            ]) for _ in range(self.numbirds)
        ])

    #Function that performs observations/ identifies neighbors
    def perform_observations(self, max_neighbours=M):
        tree = KDTree(self.positions)
        self.collisions = np.zeros(self.numbirds, dtype=int)
        new_obs = []
        for i in range(self.numbirds):
            neighbours_inds = tree.query_ball_point(self.positions[i], self.observation_radius)
            neighbours_inds.remove(i)
            neighbours = {n: 0 for n in DIRS_INDS}
            for n in neighbours_inds:
                neighbours[self.dirs[n]] += 1
            for dir in neighbours.keys():
                if neighbours[dir] > max_neighbours:
                    neighbours[dir] = max_neighbours
            self.observations[i] = neighbours

            close_birds = tree.query_ball_point(self.positions[i],self.collision_distance)
            if i in close_birds:
               close_birds.remove(i)
            self.collisions[i] = len(close_birds)

    #Function that assigns birds next state and action
    def perform_step(self):
        for i in range(self.numbirds):
            if self.actions[i] == 'V':
                i_dir = discrete_Vicsek(self.observations[i])
                if i_dir != NO_DIRS:
                    self.dirs[i] = i_dir
            elif self.actions[i] == 'I':
                self.dirs[i] = CARD_DIRS[self.instincts[i]]
            elif self.actions[i] in ['N', 'E', 'S', 'W']:
                self.dirs[i] = CARD_DIRS[self.actions[i]]
            elif self.actions[i] == 'R':
                self.dirs[i] = choice(DIRS_INDS)
            else:
                raise ValueError(f'Action {self.actions[i]} does not exist')
            self.positions[i] += STEPS[self.dirs[i]]

    #Function that assigns rewards and penalizes with a cost if two birds collide/ are within close proximity
    def reward(self, i):
        base_reward = (
            self.reward_signal * STEPS[self.dirs[i]][0]
            if self.gradient
            else self.reward_signal if self.dirs[i] == CARD_DIRS['E']
            else 0
        )
        collision_penalty = -5.0 if self.is_in_collision(i) else 0.0
        return base_reward + collision_penalty

    #Alternative to Q-learning ---> we won't use
    def Ried_learning(self):
        for i in range(self.numbirds):
            s = ternary(self.prev_obs[i].values())
            reward = self.reward(i)
            if reward:
                a_ind = self.action_space.index(self.actions[i])
                self.policies[i, s, a_ind] += reward
                self.policies[i, s] = self.policies[i, s] / sum(self.policies[i, s])

    #Defining the Q-learning algorithm
    def Q_learning(self):
        for i in range(self.numbirds):
            s = ternary(self.prev_obs[i].values())
            s_prime = ternary(self.observations[i].values())
            self.Qs[i].update(s, self.actions[i], s_prime, self.reward(i))
            argmax_Q = argmax(self.Qs[i].table[s])
            self.policies[i, s] = (
                np.array([
                    (1 - self.epsilon if j == argmax_Q else 0)
                    for j in range(len(self.action_space))
                ])
                + self.epsilon / len(self.action_space)
            )

    #Function that calculates the normalized velocity vector
    def calc_v(self):
        return sum(STEPS[dir] for dir in self.dirs) / self.numbirds

    #Function that calculates delta - the normalized distance between the learned Q tables and the ideal predefined policy
    def calc_Delta(self, print_Delta=True):
        if self.learning_alg not in ['Q', 'pol_from_Q'] or self.action_space != ['V', 'I']:
            return None
        Delta = 0
        for i in range(self.numbirds):
            desired_ind = 1 if self.instincts[i] == 'E' else 0
            if self.learning_alg == 'Q':
                Delta += np.sum(
                    self.Qs[i].table[:, desired_ind] - self.Qs[i].table[:, 1 - desired_ind] < 0
                )
            elif self.learning_alg == 'pol_from_Q':
                Delta += np.sum(
                    self.Q_tables[i, :, desired_ind] - self.Q_tables[i, :, 1 - desired_ind] < 0
                )
        Delta /= self.numbirds * len(self.observation_space)
        print(f'Delta = {Delta}')
        return Delta

    #Function to go to new state
    def update(self, tstep):
        self.actions = [choices(
            self.action_space,
            weights=self.policies[i, ternary(self.observations[i].values())]
        )[0] for i in range(self.numbirds)]

        self.perform_step()
        self.prev_obs = self.observations
        self.perform_observations()

        if self.eps_decr:
            if tstep < self.eps_decr and tstep > 0:
                self.epsilon -= self.eps_step
            elif tstep == self.eps_decr:
                self.epsilon = 0
                self.alpha = 0
                print('Learning parameters switched off')

        if self.learning_alg == 'Ried':
            self.Ried_learning()
        elif self.learning_alg == 'Q':
            self.Q_learning()

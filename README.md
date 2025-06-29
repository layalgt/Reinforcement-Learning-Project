This is the repository corresponding to the Reinforcement Learning course project. 

The project, "Studying the Flockig of Birds from a Multi-Agent Reinforcement Learning (MARL) Perspective", uses the Q-learning algorithm in an MARL framework to explore whether it is capable of reproducing emergent flocking behaviors, such as alignment, in a collective of birds.

The work builds heavily on [andredelft/flock-learning](https://github.com/andredelft/flock-learning) and introduces a slight modification to the original codebase. Namely, we add a cost associated with bird collisions and compare the resulting behavior with that obtained in the absence of penalty.


The classes [birds](birds.py) and [field](field.py) have been updated in order to include the effect of the cost mentioned above. The code used and the generated data and plots have been organized in separate folders for convenience. They can be accessed here respectively:
[without cost](https://github.com/layalgt/Reinforcement-Learning-Project/tree/c3dff627de1ec89de4f970ba6b436d804fa5e72f/Without%20Cost) and [with cost](https://github.com/layalgt/Reinforcement-Learning-Project/tree/c3dff627de1ec89de4f970ba6b436d804fa5e72f/With%20Cost).


Note on the use of AI: ChatGPT was used to assist with technical commands related to creating ZIP files for saving generated results and for adjusting image scaling parameters for plotting purposes.

ICTP, QLS, June 2025

This is the repository corresponding to the Reinforcement Learning course project. 

The project, "Studying the Flockig of Birds from a Multi-Agenet Reinforcement Learning (MARL) Perspective", uses the Q-learning algorithm in an MARL framework to explore whether it is capable of reproducing emergent flocking behaviors, such as alignment, in a collective of birds.

The work builds heavily on [andredelft/flock-learning](https://github.com/andredelft/flock-learning) and introduces a slight modification to the original codebase. Namely, we add a cost associated with bird collisions and compare the resulting behavior with that obtained in the absence of penalty.


The classes [birds](main/birds.py) and [field](main/field.py) have been updated in order to include the cost for collision (which was not present in the original codebae).


Note on the use of AI: ChatGPT was used to assist with technical commands related to creating ZIP files for saving generated results and for adjusting image scaling parameters for plotting purposes.

ICTP, QLS, June 2025

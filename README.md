# EPPO and MAPLE
In this repository we explore EPPO and MAPLE both relatively novel implementations of meta learning in reinforcement learning. This research direction tries to tackle the common dificulties related with trying to train an agent in sparse reward environments, like sokoban.
What we try:
    -Baseline check with CNN and Att feature extractors (With CNN working better at a less parameterized curriculum agenda but strugling to generalize to harder levels)
    -Recurrent policy tests, with little succes (barely any diferences found relative to non recurrent policies)
    -Maple implementation, that works with most success. This setup succesfully managed to improve the generalization capability of a sokoban  agent accustomed to 6x6 grids into playing several 10x10 grids. (Although this training was very slow which made it unpractical)
    -EPPO implementation, figure 2 in the technical report explains roughly the algorithm but in simple terms what it does is to generate multiple environments at the same time each with a pool of levels that can be sampled at random. after a play you change levels (generally there are more levels than waht the buffer supports) and set a limit of the times that the level is played. then each level in a pool is asigned a prefix that is optimized throught the epochs. The description to the  algorithm in more detail can be found in the technical report.

What works:
    -Baselines (Non recurrent)
    -Maple (Although very slow)

What doesn't work:
    -EPPO

Why do we think EPPO doesnt work: EPPO feelt to us very principled during development because it does essentially the same that maple does but faster. But we missed a critical point: flag-pole movement in gradient descent. The prefixes get optimized for helping a given state of the parameters predict the values and reduce the advantage regret of a given path. but the next time this prefix is used this same set of prefixes is no longer helping the same set of parameters (presumably unredable for the agent) and is not for the same trayectory. converting the prefixes very quickly into noise. this noise even deteriorates the learning schedule making the agent even harder to learn.


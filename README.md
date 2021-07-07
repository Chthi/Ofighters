# Ofighters

Spaceship combat simulator made as an artificial intelligence arena.
I'm playing with reinforcement learning, dense and convolutional models with multiple inputs and outputs and neural network made from scratch.

Screenshots and captures might differ from actual version of the simulator but reflect the overall idea of the interface.

![menu_demo](images/menu_demo.gif)


Usage :
```
python ofighters.py
```

## Installation
With conda
```
conda env create -f requirements.yml -n ofighters
conda activate ofighters
```

## Features :

![ofighters_presentation](images/ofighters_presentation.gif)

##### Player transfer
A ship can be directly controlled by and human by clicking the ```transfert player``` button. The player ship is gray and shoot green lasers.

![random_bot_fight_player](images/random_bot_fight_player.gif)

##### Continuous training
Restart the game every 200 iterations.
The number of iterations can be changed in ```ofighters.py``` by changing the constant ```MAX_TIME```.

##### Neural network based artificial intelligence
Many bots are available, but the main one is based on a combination of convolutional and dense layers with two inputs (2D and 1D) and 2 outputs (2D cursor and 1D action vector). It also uses reinforcement learning to study its past actions and to improve, but it does not actually converge and become good (still fun to watch).

![simple_untrained_network](images/simple_untrained_network.gif)

##### Loss graph
Loss graph display the loss of the model used for the bot.
Allows to keep track of progression.

![loss_graph](images/loss_bi_head_pointer_replay_300.png)

##### Score graph
Score graph display the total score of the bot for each game.
Useful to keep track of the progress on the long run.

![score_graph](images/score_example.png)

##### Exploration rate (epsilon)
There is an exploration rate that define the chances of random actions.
Epsilon graph display this rate against time.
The rate change is automated and can vary linearly or sinusoidally.

![epsilon_graph](images/sine_epsilon_5000.png)

##### Action map
The action map displays the decision  weights at the output of the model.
The bot takes decision based on them.

![action_map](images/action_map_5000.png)

##### Bot inputs
On top of the action map we can see some of the inputs fed to the bot.

![bot_inputs](images/bot_inputs.png)


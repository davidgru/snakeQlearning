
# Idea
  * Use Deep-Q-Learning to train an agent to play snake.
  * Q-function is approximated by neural network
  * Input state to the Q-function is the whole 10x10 grid

# Notes

## Network
  * Use CNN to take advantage of spatial structure
  * Without padding, snake struggles to see food at the edge
  * Unsure about size

## Exploration Strategy
  * Epsilon-Greedy doesn't work well because it samples actions that lead to certain death, therefore snake doesn't get long 
  * Softmax works better
  * TODO: Find good temperature decay rate

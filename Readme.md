
# Idea
  * Train an agent to play snake
  * Give agent full view of the board and let him figure out key features himself
  * Q-function is approximated by neural network
  * Use target network and replay memory

# How to run this?
1. Create virtual environment.
```
python3 -m venv env
```
2. Activate it.
```
source env/bin/activate
```
3. Install required libraries.
```
pip install -r requirements.txt
```
4. Test pretrained model
```
python test.py model.pt
```
5. Train yourself
```
python main.py
```

# MDP

## State Space
A state is made of a `10x10` grid where:
  * `empty` is represented by `-0.1`
  * `body` is represented by `1`
  * `head` is represented by `2`
  * `food` is represented by `3`

## Action Space
There are 4 actions:
  * `left` represented by `0`
  * `right` represented by `1`
  * `up` represented by `2`
  * `down` represented by `3`

## Reward Function
  * `r(s,a) = -2`, if the action results in a crash with wall or body
  * `r(s,a) = 1`, if the agent ate food
  * `r(s,a) = -0.01`, in any other scenario

# Notes

## Network
  * Use CNN to take advantage of spatial structure
  * Without padding, snake struggles to see food at the edge
  * Unsure about size

## Exploration Strategy
  * Epsilon-Greedy doesn't work well because it samples actions that lead to certain death, therefore snake doesn't get long 
  * Softmax works better
  * TODO: Find good temperature decay rate

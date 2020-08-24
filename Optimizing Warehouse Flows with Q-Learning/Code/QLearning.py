#artificial inteligence for business
#optimizing warehouse flows with Q-learning

import numpy as np


def route(starting_location, ending_location):

    R = np.copy(pR)

    ending_state = location_to_state[ending_location]
    R[ending_state, ending_state] = 1000

    #BELLMAN EQUATION FOR TRAINING Q MATRIX
    Q = np.array(np.zeros([12,12]))
    for i in range(1000):
        current_state = np.random.randint(0,12)
        playableActions = []

        for j in range(12):
            if R[current_state,j] > 0:
                playableActions.append(j)
        next_state = np.random.choice(playableActions)
        TD = R[current_state, next_state] + gama * Q[next_state, np.argmax(Q[next_state, ])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD


    #going into production

    #making mapping from states to location

    route = [starting_location]
    next_location = starting_location

    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route


    print(Q.astype(int))


def full_route_quick(start,inter,end): #gets full route assuming user knows which point is the mid point and the end point
    return route(start,inter) + route(inter,end)[1:]

if __name__ == '__main__':

    #dictate speed of q learning
    gama = .75 #temporal difference minimizing
    alpha = .9 #distance punishment

    #defining environment
    location_to_state = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11 }
    state_to_location = {state: location for location, state in location_to_state.items()}
    actions = [0,1,2,3,4,5,6,7,8,9,10,11]

    # Defining the rewards, see picture of map for physical image of the space matrix represents
    pR = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

    print(full_route_quick('E','F','G'))
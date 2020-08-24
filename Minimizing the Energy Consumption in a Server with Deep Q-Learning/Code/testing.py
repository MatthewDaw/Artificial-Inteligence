


import os
import numpy as np
import random as rn
import environment
from keras.models import load_model

if __name__ == '__main__':

    #SETTIN RANDOM SEEDS FOR DEBUGGING PURPOSES
    os.environ['YTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)

    number_actions = 5
    direction_boundry = (number_actions - 1) / 2
    temperature_step = 1.5

    env = environment.Environment(optimal_temperature = (10.0,14.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

    #LOADING A PRE TRAINED MODEL
    model = load_model("model.h5")

    train = False
    env.train = train

    env.train = False

    current_state, _, _ = env.observe()

    #SUMMING ENERGY EXPENDED FOR BOTH  SIMULATIONS
    for timestep in range(0, 12*30):
        q_values = model.predict(current_state)
        action = np.argmax(q_values[0])
        if (action - direction_boundry < 0):
            direction = -1
        else:
            direction = 1
        energy_ai = abs(action - direction_boundry) * temperature_step

        next_state, _, _ = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
        current_state = next_state

    #PRINTING RESULTS
    print()
    print("total energy with an AI: {:.0f}".format(env.total_energy_ai))
    print("total energy with no AI: {:.0f}".format(env.total_energy_noai))
    print("Energy saved: {:.0f} %".format(((env.total_energy_noai - env.total_energy_ai) / (env.total_energy_noai))*100))

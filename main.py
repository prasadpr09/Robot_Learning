import numpy as np
import matplotlib.pyplot as plt
from histogram_filter_copy import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('C:/Users/prkri/OneDrive/Desktop/spring/learning in robotics/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    print("belief_states: \n", belief_states)
    # print(belief_states.shape)
    # print("cmap:", cmap)
    # print("actions", actions)
    # print("observations", observations)

    #### Test your code here

    filter = HistogramFilter()
    belief = np.ones((cmap.shape[0], cmap.shape[1]))  # divide it 

    # print(belief)
    for i in range(actions.shape[0]):
        belief = filter.histogram_filter(cmap, belief, actions[i], observations[i])
        belief_rot = np.rot90(belief, -1)
        max_index = np.unravel_index(np.argmax(belief_rot), belief_rot.shape)
        print(max_index)
    

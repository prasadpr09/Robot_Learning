import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''
        # belief_states is just to confirm - this isnt used. 
        # cmap is the color map (ground truth)
        # action gives us where to go, that is N,S,E,W
        # observation is from color sesnsor

        belief = np.rot90(belief, -1)
        cmap = np.rot90(cmap, -1)

        rows,col = belief.shape
        new_belief = np.zeros((rows, col))  # Initialize new belief distribution

        for r in range(rows):
            for c in range(col):

                # moved to new location
                new_r, new_c = r + action[0], c + action[1]
                
                if 0 <= new_r < rows and 0 <= new_c < col:
                    new_belief[new_r, new_c] = 0.9 * belief[r, c] + 0.1 * belief[r, c]
                else:
                    new_belief[r,c] = 0.1 * belief[r, c]


        # for i in range(rows):
        #     for j in range(col):
        #         if cmap[i,j] == observation:
        #             new_belief[i,j]  *= 0.9 
        #         else:
        #             new_belief[i,j]  *= 0.1 

        a = np.where(cmap == observation, 0.9, 0.1 )
        new_belief1 = new_belief * a
        

        # normalize it 
        new_belief1 /= np.sum(new_belief1)

        new_belief1 = np.rot90(new_belief1,1)

        return new_belief1
    



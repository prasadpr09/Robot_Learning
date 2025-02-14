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
        belief = np.rot90(belief, -1)
        cmap = np.rot90(cmap, -1)

        row,col = belief.shape
        new_belief_matrix = np.zeros((row,col))

        prev_x = action[0] 
        prev_y = action[1]

        for i in range(0,row):
            for j in range(0,col):

                [i_,j_] = [i,j] - action

                if 0 <= i_ < row and 0 <= j_ < col:
                    new_belief_matrix[i,j] = 0.1 *  belief[i,j] + 0.9 * belief[i_, j_]

                else: 
                    new_belief_matrix[i,j] = 0.1 *  belief[i,j]

        
        a = np.where(cmap == observation, 0.9, 0.1 )
        posterior = new_belief_matrix * a
        
        # normalize it 
        posterior /= np.sum(posterior)


        posterior = np.rot90(posterior,1)

        return posterior
    

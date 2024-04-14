import lap
import numpy as np


def main(cost, thresh=0.5):
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    print(cost, x, y) 



if __name__ == '__main__':
    cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    main(cost_matrix)  


    cost_matrix = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
    main(cost_matrix)  
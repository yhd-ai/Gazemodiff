from scipy.stats import wilcoxon
import numpy as np

proposedList = np.load('error_gimo_proposed.npy')
humanmacList = np.load('error_gimo_humanmac.npy')




res = wilcoxon(proposedList,humanmacList)
print(res)
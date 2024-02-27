import numpy as np

normMean = [0.37225845, 0.38512826, 0.38392654]
normStd = [0.20772879, 0.20576294, 0.21414135]

Mean = [np.round(i*225,2) for i in normMean]
Std =  [np.round(j*225,2) for j in normStd]

print("Mean is ",Mean)
print("Std is ", Std)

'''
Mean is  [83.76, 86.65, 86.38]
Std is  [46.74, 46.3, 48.18]
'''
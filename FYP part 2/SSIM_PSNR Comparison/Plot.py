import matplotlib.pyplot as plt
import numpy as np

psnre = [27.862212224481663, 27.77568044128489, 28.238496432126677, 27.96169514253027, 
27.870256363718468, 27.910003702005763, 27.90734028040077, 28.039473609443366, 27.931600621363827]

psnro = [27.833058370828695, 27.752582631782722, 28.113412759598724, 28.005508987161033, 
27.87001517894729, 27.909294040622918, 27.981278500745688, 27.944538159681716, 27.945613920923815]

X = ['1','2','3','4','5','6','7','8','9']

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2 , psnro , 0.4 , label = 'Our Model')
plt.bar(X_axis + 0.2 , psnre , 0.4 , label = 'ESRGAN')
  
plt.xticks(X_axis, X)
plt.xlabel("Each Test Image")
plt.ylabel("PSNR Value")
plt.legend()
plt.show()
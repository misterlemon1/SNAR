import numpy as np

from HW6.QUAT import Quaternion

q=Quaternion([0,np.sqrt(3),1,0])
print(q.rotate_deg([0,0,1],420))

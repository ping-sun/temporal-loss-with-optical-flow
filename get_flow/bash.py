#p=sub.Popen(['bash','makeOptFlow.sh','05/%05d.jpg','05/res/','0','5'])
#p.wait()

import os
import glob 
import time
import subprocess as sub

num = 0
save_flow_file = '/save/path/'
image_path = '/image/path/'
total = len(os.listdir(save_flow_file))

for f in glob.glob(image_path + "*"):
  fname = f.split("/")[-1]
  save_path = save_flow_file+fname
  if not os.path.exists(save_path):
    os.mkdir(save_path)
    
  t1 = time.time()
  num+=1
  print(f"({num}/{total})  {fname}")

  p=sub.Popen(['bash','makeOptFlow.sh', f+'/%04d.png', save_path, '0', '1'])  #'0'
  p.wait()

  print("time: %.2f s "%float(time.time()-t1))
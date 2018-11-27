import pickle
import numpy as np
import gzip
import os
import glob
import time

def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height,
                                                                   width, 2))
    f.close()
    return flow.astype(np.int16)

def topickle(flow, name, file_dir):
    pickle.dump(flow,open(f'{file_dir}/{name}.pkl',"wb"))
    gzip.GzipFile(f'{file_dir}/{name}.pkl','wb',compresslevel=3).write(pickle.dumps(flow))

def main():
    print("Begin processing...")
    print(os.getcwd())
    t0 = time.time()
    for f in glob.glob("*"):
        t1 = time.time()
        print(f"processing file: {f}")
        file_dir = 'file_dir/' + f
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        
        for flo in glob.glob(os.path.join(f, "*.flo")):
            fname = flo.split("/")[-1].split('.')[0]
            #print(fname)
            flow = readFlow(flo)
            topickle(flow, fname, file_dir)
        print(f"time: {time.time()-t1:.2f}")
    print(f"total time: {time.time()-t0:.2f}")

if __name__ == "__main__":
    main()
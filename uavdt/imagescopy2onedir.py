import os
import shutil


multidirpath = './UAV-benchmark-M-test'
outdir = './UAV-M-test'
# os.mkdir((outdir))

filenames = os.listdir(multidirpath)
# print(filenames)
for file in filenames:
    wholefile = multidirpath + '/' + file
    filenextname = os.listdir(wholefile)
    # print(filenextname)
    for filenext in filenextname:
        pathall = multidirpath + '/' + file + '/' + filenext
        # print(pathall)
        str1 = str(file)
        outpath = outdir + '/' + str1 + '_' + filenext[-10:]
        # print(outpath)
        shutil.copyfile(pathall,outpath)
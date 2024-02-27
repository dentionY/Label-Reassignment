2023-07-31使用说明：
1、总共的图片有40735张，不区分train、valid、test；查阅论文大多数处理方案是23258张train、15069张test，然而加起来只有38327张。
2、因此打算，按照片段作为基本单位，随机划分。近似按照15:1:4(train、valid、test)划分50个片段，因此train有35个、valid有5个、test有10个。
3、图片数量如下：
(base) root@autodl-container-c8da118cfa-3261e683:~/autodl-tmp/uavdt/UAV-M-test# ls -l|grep "-"| wc -l
8588
(base) root@autodl-container-c8da118cfa-3261e683:~/autodl-tmp/uavdt/UAV-M-test# cd ../UAV-M-train
(base) root@autodl-container-c8da118cfa-3261e683:~/autodl-tmp/uavdt/UAV-M-train# ls -l|grep "-"| wc -l
30444
(base) root@autodl-container-c8da118cfa-3261e683:~/autodl-tmp/uavdt/UAV-M-train# cd ../UAV-M-val
(base) root@autodl-container-c8da118cfa-3261e683:~/autodl-tmp/uavdt/UAV-M-val# ls -l|grep "-"| wc -l
1703

.py文件说明：
1、draw_pic.py：用于得到json文件后对原图画出带框的图；
2、images2all.py：将二级目录下的所有图片文件全部挪移到一级目录下；
3、imagescopy2onedir.py：作用同2
4、JsonAllocate.py：使用UAV-new.json根据已经分配好图片的train、val、test文件夹进行json分配，得到各自的json；
5、numpic.py：数图片数量；
6、ShuffleEpisode.py：用于随机划分图片的epsiode，并分配到train、val、test；
7、txt2json.py：对2得到的结果进行json整理，得到UAV-new.json；
8、mean_std.py：用于计算图片集RGB的均值和方差；


必要的shell命令：
1、# 查看当前目录下的文件数量（不包含子目录中的文件）：ls -l|grep "^-"| wc -l
2、# 查看当前目录下的文件夹目录个数（不包含子目录中的目录），同上述理，如果需要查看子目录的，加上R：ls -l|grep "^d"| wc -l
3、# 查看当前目录下的携带某关键字的文件数量：ls -l|grep "M010"| wc -l


均值和方差：
Mean is  [83.75815125, 86.6538585, 86.3834715]
Std is  [46.73897775, 46.2966615, 48.18180375]

normalize: [[83.76, 86.65, 86.38], [46.74, 46.3, 48.18]] #uavdt
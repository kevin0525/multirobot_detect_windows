//-----------------------宏定义----------------------------
//---------------------------------------------------------
#define IrobotSetNo 313			//机器人样本个数  
#define ObstacleSetNo 174		//障碍样本个数
#define BackgroundSetNo 499		//背景样本个数
#define HardBackgroundSetNo 18	//Hard背景样本个数

#define SHOWSET false			//是否显示训练样本
#define TRAIN true				//是否进行训练,true表示训练，false表示读取xml文件中的SVM模型
#define SAVESET true			//是否保存检测数据

/*//for cpu
//HOG描述子参数
#define WinSizeDetect Size(32,16)		//检测窗口尺寸
#define BlockSizeDetect Size(8,8)		//block尺寸
#define BlockStrideDetect Size(4,4)		//block步长
#define CellSizeDetect Size(4,4)		//cell尺寸
#define NbinsDetect 9					//直方图bin个数

//HOG描述子参数
#define WinSizeClassify Size(48,24)		//检测窗口尺寸
#define BlockSizeClassify Size(16,8)		//block尺寸
#define BlockStrideClassify Size(4,4)	//block步长
#define CellSizeClassify Size(4,4)		//cell尺寸
#define NbinsClassify 9					//直方图bin个数
*/

///////////for GPU
//HOG描述子参数
#define WinSizeDetect Size(40,40)		//检测窗口尺寸
#define BlockSizeDetect Size(16,16)		//block尺寸
#define BlockStrideDetect Size(8,8)		//block步长
#define CellSizeDetect Size(8,8)		//cell尺寸
#define NbinsDetect 9					//直方图bin个数

//HOG描述子参数
#define WinSizeClassify Size(40,40)		//检测窗口尺寸
#define BlockSizeClassify Size(16,16)		//block尺寸
#define BlockStrideClassify Size(8,8)	//block步长
#define CellSizeClassify Size(8,8)		//cell尺寸
#define NbinsClassify 9					//直方图bin个数

////HOG描述子参数
//#define WinSize Size(20,10)		//检测窗口尺寸
//#define BlockSize Size(4,4)		//block尺寸
//#define BlockStride Size(2,2)	//block步长
//#define CellSize Size(2,2)		//cell尺寸
//#define Nbins 9					//直方图bin个数

////HOG描述子参数
//#define WinSize Size(64,32)		//检测窗口尺寸
//#define BlockSize Size(8,8)		//block尺寸
//#define BlockStride Size(4,4)		//block步长
//#define CellSize Size(4,4)		//cell尺寸
//#define Nbins 9					//直方图bin个数

//detectMultiScale部分参数
#define HitThreshold 0			//特征向量与超平面最小距离
#define WinStride Size(8,8)		//移动步长(必须是block步长的整数倍)
#define DetScale 1.1			//源图像图像每次缩小比例

#define TestImage "../Data/TestImage/13.jpg"				//用于检测的测试图像
#define ResultImage "../Data/Result/13.jpg"					//测试图像的检测结果
#define ResultImageFile_1 "..\\Data\\Result\\13-1\\"		//测试图像的分类框图文件夹1
#define ResultImageFile_2 "..\\Data\\Result\\13-2\\"		//测试图像的分类框图文件夹2
#define ResultImageFile_3 "..\\Data\\Result\\13-3\\"		//测试图像的分类框图文件夹3
#define TestVideo "../Data/TestVideo/329_2.avi"			//用于检测的测试视频
#define ResultVideo "../Data/Result/329_2.avi"			//测试视频的检测结果
#define ResultVideoFile_1 "..\\Data\\Result\\329_1\\"	//测试视频的分类框图文件夹1
#define ResultVideoFile_2 "..\\Data\\Result\\329_2\\"	//测试视频的分类框图文件夹2
#define ResultVideoFile_3 "..\\Data\\Result\\329_3\\"	//测试视频的分类框图文件夹3

#define IrobotSetFile "../Data/IrobotSet/"					//机器人样本图片文件夹
#define ObstacleSetFile "../Data/ObstacleSet/"				//障碍样本图片文件夹
#define BackgroundSetFile "../Data/BackgroundSet/"			//背景样本图片文件夹
#define HardBackgroundSetFile "../Data/HardBackgroundSet/"	//Hard样本图片文件夹
#define SetName "0SetName.txt"								//样本图片的文件名列表txt
#define DetectSvmName "../Data/Result/SVM_HOG_Detect.xml"	//保存与读取的检测模型文件名称
#define ClassifySvmName "../Data/Result/SVM_HOG_Classify.xml"//保存与读取的分类模型文件名称


#define TrainPerc 1		//训练比例
#define VaildPerc 0		//交叉验证比例
//#define TestPerc 0.2
#define IrobotTrainNo (int(IrobotSetNo * TrainPerc))//训练样本数量
#define ObstacleTrainNo (int(ObstacleSetNo * TrainPerc))
#define BackgroundTrainNo (int(BackgroundSetNo * TrainPerc))
#define HardBackgroundTrainNo (int(HardBackgroundSetNo * TrainPerc))
#define AllTrainNo (IrobotTrainNo + ObstacleTrainNo + BackgroundTrainNo + HardBackgroundTrainNo)
#define IrobotVaildNo (int(IrobotSetNo * VaildPerc))//交叉验证样本数量
#define ObstacleVaildNo (int(ObstacleSetNo * VaildPerc))
#define BackgroundVaildNo (int(BackgroundSetNo * VaildPerc))
#define HardBackgroundVaildNo (int(HardBackgroundSetNo * VaildPerc))
#define AllVaildNo (IrobotVaildNo + ObstacleVaildNo + BackgroundVaildNo + HardBackgroundVaildNo)
#define IrobotTestNo (IrobotSetNo - IrobotTrainNo - IrobotVaildNo)//测试样本数量
#define ObstacleTestNo (ObstacleSetNo - ObstacleTrainNo - ObstacleVaildNo)
#define BackgroundTestNo (BackgroundSetNo - BackgroundTrainNo - BackgroundVaildNo)
#define HardBackgroundTestNo (HardBackgroundSetNo - HardBackgroundTrainNo - HardBackgroundVaildNo)
#define AllTestNo (IrobotSetNo + ObstacleSetNo + BackgroundSetNo + HardBackgroundSetNo)

//#include <iostream>  
//#include <fstream>  
//#include <strstream>
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>  
//#include <opencv2/imgproc/imgproc.hpp>  
//#include <opencv2/objdetect/objdetect.hpp>  
//#include <opencv2/ml/ml.hpp>  
//
//using namespace std;  
//using namespace cv;  
//
////-----------------------宏定义----------------------------
////---------------------------------------------------------
//#define IrobotSetNo 218			//机器人样本个数  
//#define ObstacleSetNo 205		//障碍样本个数
//#define BackgroundSetNo 736		//背景样本个数
//#define HardBackgroundSetNo 155	//Hard背景样本个数
//
//#define SHOWSET false			//是否显示训练样本
//#define TRAIN false				//是否进行训练,true表示训练，false表示读取xml文件中的SVM模型
//#define SAVESET false			//是否保存检测数据
//
////HOG描述子参数
//#define WinSize Size(40,20)		//检测窗口尺寸
//#define BlockSize Size(8,8)		//block尺寸
//#define BlockStride Size(4,4)	//block步长
//#define CellSize Size(4,4)		//cell尺寸
//#define Nbins 9					//直方图bin个数
//
//////HOG描述子参数
////#define WinSize Size(32,16)		//检测窗口尺寸
////#define BlockSize Size(8,8)		//block尺寸
////#define BlockStride Size(4,4)	//block步长
////#define CellSize Size(4,4)		//cell尺寸
////#define Nbins 9					//直方图bin个数
//
//////HOG描述子参数
////#define WinSize Size(20,10)		//检测窗口尺寸
////#define BlockSize Size(4,4)		//block尺寸
////#define BlockStride Size(2,2)	//block步长
////#define CellSize Size(2,2)		//cell尺寸
////#define Nbins 9					//直方图bin个数
//
//////HOG描述子参数
////#define WinSize Size(64,32)		//检测窗口尺寸
////#define BlockSize Size(8,8)		//block尺寸
////#define BlockStride Size(4,4)		//block步长
////#define CellSize Size(4,4)		//cell尺寸
////#define Nbins 9					//直方图bin个数
//
////detectMultiScale部分参数
//#define HitThreshold 0			//特征向量与超平面最小距离
//#define WinStride Size(4,4)		//移动步长(必须是block步长的整数倍)
//#define DetScale 1.05			//源图像图像每次缩小比例
//
//#define TestImage "../Data/TestImage/13.jpg"				//用于检测的测试图像
//#define ResultImage "../Data/Result/13.jpg"					//测试图像的检测结果
//#define ResultImageFile_1 "..\\Data\\Result\\13-1\\"		//测试图像的分类框图文件夹1
//#define ResultImageFile_2 "..\\Data\\Result\\13-2\\"		//测试图像的分类框图文件夹2
//#define ResultImageFile_3 "..\\Data\\Result\\13-3\\"		//测试图像的分类框图文件夹3
//#define TestVideo "../Data/TestVideo/1轮-浙大.avi"			//用于检测的测试视频
//#define ResultVideo "../Data/Result/1轮-浙大.avi"			//测试视频的检测结果
//#define ResultVideoFile_1 "..\\Data\\Result\\1轮-浙大-1\\"	//测试视频的分类框图文件夹1
//#define ResultVideoFile_2 "..\\Data\\Result\\1轮-浙大-2\\"	//测试视频的分类框图文件夹2
//#define ResultVideoFile_3 "..\\Data\\Result\\1轮-浙大-3\\"	//测试视频的分类框图文件夹3
//
//#define IrobotSetFile "../Data/IrobotSet/"					//机器人样本图片文件夹
//#define ObstacleSetFile "../Data/ObstacleSet/"				//障碍样本图片文件夹
//#define BackgroundSetFile "../Data/BackgroundSet/"			//背景样本图片文件夹
//#define HardBackgroundSetFile "../Data/HardBackgroundSet/"	//Hard样本图片文件夹
//#define SetName "0SetName.txt"								//样本图片的文件名列表txt
//
//#define DetectSvmName "../Data/Result/SVM_HOG_Detect.xml"	//保存与读取的检测模型文件名称
//#define ClassifySvmName "../Data/Result/SVM_HOG_Classify.xml"//保存与读取的分类模型文件名称
//
////-----------------------主函数----------------------------
////---------------------------------------------------------
//
//int main()  
//{
//	//变量定义
//    HOGDescriptor hog(WinSize,BlockSize,BlockStride,CellSize,Nbins);//HOG描述子：检测窗口，block尺寸，block步长，cell尺寸，直方图bin个数 
//    int descriptorDim;//HOG描述子的维数：[(检测窗口长-block长)/block步长+1]*[(检测窗口高-block高)/block步高+1]*cell长*cell高*bin个数
//    MySVM detectSvm;//检测SVM
//	MySVM classifySvm;//分类SVM
//
//    //----------------训练分类器or直接读取分类器---------------------
//	//---------------------------------------------------------------
//    if(TRAIN) //训练分类器，并保存XML文件
//    {
//		//训练变量定义
//        string ImgName;//图片名
//        ifstream IrobotName((string)IrobotSetFile+SetName);//机器人样本图片的文件名列表
//		ifstream ObstacleName((string)ObstacleSetFile+SetName);//障碍样本图片的文件名列表
//        ifstream BackgroundName((string)BackgroundSetFile+SetName);//背景样本图片的文件名列表
//		ifstream HardBackgroundName((string)HardBackgroundSetFile+SetName);//Hard背景样本图片的文件名列表 
//        Mat sampleFeatureMat;//训练SVM的特征向量矩阵：行数=样本个数，列数=特征向量维数
//        Mat detectLabelMat;//检测SVM的的类别向量：行数=样本个数，列数=1：1表示有机器人或障碍，-1表示无机器人和障碍
//		Mat classifyLabelMat;//分类SVM的的类别向量：行数=样本个数，列数=1：1表示有机器人，2表示有障碍，3表示有背景
//  
//        //1.处理机器人样本图片
//        for(int num=0; num<IrobotSetNo && getline(IrobotName,ImgName); num++)  
//        {  
//            ImgName = IrobotSetFile + ImgName;//加上机器人样本的路径名  
//			cout<<"处理："<<ImgName<<endl;  
//            Mat src = imread(ImgName);//读取图片  
//			resize(src,src,WinSize);//将训练样本归一化为检测窗口的大小
//			if(SHOWSET)//是否显示训练样本
//			{
//				imshow("【机器人样本】",src);
//				if (waitKey(1)>0){return 0;}//通过按键中断程序
//			}
//			
//            vector<float> descriptors;//HOG描述子向量  
//            hog.compute(src,descriptors);//计算HOG描述子
//  
//            if(num == 0)//处理第一个样本时初始化特征向量矩阵和类别矩阵  
//            {
//                descriptorDim = descriptors.size();//HOG描述子的维数  
//                sampleFeatureMat = Mat::zeros(IrobotSetNo+ObstacleSetNo+BackgroundSetNo+HardBackgroundSetNo, descriptorDim, CV_32FC1);  
//                detectLabelMat = Mat::zeros(IrobotSetNo+ObstacleSetNo+BackgroundSetNo+HardBackgroundSetNo, 1, CV_32FC1);  
//				classifyLabelMat = Mat::zeros(IrobotSetNo+ObstacleSetNo+BackgroundSetNo+HardBackgroundSetNo, 1, CV_32FC1);  
//            } 
//
//            //将计算好的HOG描述子复制到特征向量矩阵和类别矩阵
//            for(int i=0; i<descriptorDim; i++)
//                sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
//            detectLabelMat.at<float>(num,0) = 1;//1表示有机器人或障碍 
//			classifyLabelMat.at<float>(num,0) = 1;//1表示有机器人
//        }  
//
//		//2.处理障碍样本图片
//		for(int num=IrobotSetNo; num<IrobotSetNo+ObstacleSetNo && getline(ObstacleName,ImgName); num++)  
//		{ 
//			ImgName = ObstacleSetFile + ImgName;//加上障碍样本的路径名
//			cout<<"处理："<<ImgName<<endl;  
//			Mat src = imread(ImgName);//读取图片 
//			resize(src,src,WinSize);//将训练样本归一化为检测窗口的大小
//			if(SHOWSET)//是否显示训练样本
//			{
//				imshow("【背景样本】",src);
//				if (waitKey(1)>0){return 0;}//通过按键中断程序
//			}
//
//			vector<float> descriptors;//HOG描述子向量  
//			hog.compute(src,descriptors);//计算HOG描述子
//
//			//将计算好的HOG描述子复制到特征向量矩阵和类别矩阵
//			for(int i=0; i<descriptorDim; i++)  
//				sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
//			detectLabelMat.at<float>(num,0) = 1;//1表示有机器人或障碍
//			classifyLabelMat.at<float>(num,0) = 2;//2表示有障碍
//		}
//
//        //3.处理背景样本图片
//        for(int num=IrobotSetNo+ObstacleSetNo; num<IrobotSetNo+ObstacleSetNo+BackgroundSetNo && getline(BackgroundName,ImgName); num++)  
//        {  
//            ImgName = BackgroundSetFile + ImgName;//加上背景样本的路径名  
//			cout<<"处理："<<ImgName<<endl;  
//            Mat src = imread(ImgName);//读取图片 
//			resize(src,src,WinSize);//将训练样本归一化为检测窗口的大小
//			if(SHOWSET)//是否显示训练样本
//			{
//				imshow("【背景样本】",src);
//				if (waitKey(1)>0){return 0;}//通过按键中断程序
//			}
//
//            vector<float> descriptors;//HOG描述子向量  
//            hog.compute(src,descriptors);//计算HOG描述子
//  
//            //将计算好的HOG描述子复制到特征向量矩阵和类别矩阵
//            for(int i=0; i<descriptorDim; i++)  
//                sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
//            detectLabelMat.at<float>(num,0) = -1;//-1表示无机器人和障碍
//			classifyLabelMat.at<float>(num,0) = 3;//3表示有背景
//        }
//
//        //4.处理Hard背景样本图片
//        for(int num=IrobotSetNo+ObstacleSetNo+BackgroundSetNo; num<IrobotSetNo+ObstacleSetNo+BackgroundSetNo+HardBackgroundSetNo && getline(HardBackgroundName,ImgName); num++)  
//        {
//            ImgName = HardBackgroundSetFile + ImgName;//加上Hard背景样本的路径名  
//			cout<<"处理："<<ImgName<<endl;
//            Mat src = imread(ImgName);//读取图片 
//			resize(src,src,WinSize);//将训练样本归一化为检测窗口的大小
//			if(SHOWSET)//是否显示训练样本
//			{
//				imshow("【HardExample背景样本】",src);
//				if (waitKey(1)>0){return 0;}//通过按键中断程序
//			}
//
//            vector<float> descriptors;//HOG描述子向量  
//            hog.compute(src,descriptors);//计算HOG描述子
//  
//            //将计算好的HOG描述子复制到特征向量矩阵和类别矩阵 
//            for(int i=0; i<descriptorDim; i++)  
//                sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
//            detectLabelMat.at<float>(num,0) = -1;//背景样本类别为-1，无机器人
//			classifyLabelMat.at<float>(num,0) = 3;//3表示有背景
//        }  
//
//        //5.训练检测SVM分类器  
//        //迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
//        CvTermCriteria detectCriteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);  
//        //SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
//        CvSVMParams detectParam(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, detectCriteria);  
//        cout<<"开始训练检测SVM分类器"<<endl;  
//        detectSvm.train(sampleFeatureMat, detectLabelMat, Mat(), Mat(), detectParam);//训练分类器
//        cout<<"训练完成"<<endl;  
//        detectSvm.save(DetectSvmName);//将训练好的SVM模型保存为xml文件  
//
//		//6.训练分类SVM分类器  
//		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
//		CvTermCriteria classifyCriteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);  
//		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
//		CvSVMParams classifyParam(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, classifyCriteria);  
//		cout<<"开始训练分类SVM分类器"<<endl;  
//		classifySvm.train(sampleFeatureMat, classifyLabelMat, Mat(), Mat(), classifyParam);//训练分类器
//		cout<<"训练完成"<<endl;  
//		classifySvm.save(ClassifySvmName);//将训练好的SVM模型保存为xml文件  
//    }  
//    else //若TRAIN为false，从XML文件读取训练好的分类器  
//    {  
//        detectSvm.load(DetectSvmName);
//		classifySvm.load(ClassifySvmName);
//    }  
//
//	//----------------进行机器人和障碍物的检测与分类-----------------
//	//---------------------------------------------------------------
//	//变量定义
//	HOGDescriptor detectHOG(WinSize,BlockSize,BlockStride,CellSize,Nbins);//分类HOG检测器
//	HOGDescriptor classifyHOG(WinSize,BlockSize,BlockStride,CellSize,Nbins);//检测HOG：用于计算检测结果的HOG特征向量
//    descriptorDim = detectSvm.get_var_count();//特征向量的维数，即HOG描述子的维数（和前面训练时的大小一样，添加此句是为了在不训练时也能拿到维数）
//    int supportVectorNum = detectSvm.get_support_vector_count();//支持向量的个数  
//    cout<<"支持向量个数："<<supportVectorNum<<endl;  
//    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数  
//    Mat supportVectorMat = Mat::zeros(supportVectorNum, descriptorDim, CV_32FC1);//支持向量矩阵  
//    Mat resultMat = Mat::zeros(1, descriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果  
//
//    //计算w矩阵
//    for(int i=0; i<supportVectorNum; i++)//将支持向量的数据复制到supportVectorMat矩阵中  
//	{
//        const float * pSVData = detectSvm.get_support_vector(i);//返回第i个支持向量的数据指针  
//        for(int j=0; j<descriptorDim; j++)  
//            supportVectorMat.at<float>(i,j) = pSVData[j];  
//    }
//    double * pAlphaData = detectSvm.get_alpha_vector();//返回SVM的决策函数中的alpha向量  
//    for(int i=0; i<supportVectorNum; i++)//将alpha向量的数据复制到alphaMat中  
//        alphaMat.at<float>(0,i) = pAlphaData[i];  
//    resultMat = -1 * alphaMat * supportVectorMat;//计算-(alphaMat * supportVectorMat),结果放到resultMat中 
//
//    //得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子  
//    vector<float> myDetector;//基于Hog特征的SVM检测子（w+b）
//    for(int i=0; i<descriptorDim; i++)//将resultMat中的数据复制到数组myDetector中  
//        myDetector.push_back(resultMat.at<float>(0,i));  
//    myDetector.push_back(detectSvm.get_rho());//最后添加偏移量rho，得到检测子  
//    cout<<"基于Hog特征的SVM检测子维数(w+b)："<<myDetector.size()<<endl;
//
//	//设置SVMDetector检测子
//    detectHOG.setSVMDetector(myDetector);  
//
//
//
// //   //-------------------------读入图片进行检测与分类-----------------------------------
//	////变量定义
// //   Mat src = imread(TestImage); //读取被测图像
//	////resize(src,src,Size(0,0),2,2);//调整输入图像的大小
//	//Mat dst; //输出图像
//	//src.copyTo(dst);
//	//vector<Rect> found, found_filtered;//检测框容器
//	//const char * sd1 = {"md "ResultImageFile_1};//创建存放检测框图的文件夹
//	//system(sd1);
//	//const char * sd2 = {"md "ResultImageFile_2};//创建存放检测框图的文件夹
//	//system(sd2);
//	//const char * sd3 = {"md "ResultImageFile_3};//创建存放检测框图的文件夹
//	//system(sd3);
//	//
//	////对图片进行多尺度机器人检测 
// //   cout<<"进行多尺度检测"<<endl;
// //   detectHOG.detectMultiScale(src, found, HitThreshold, WinStride, Size(0,0), DetScale, 2, false);
//	//	//参数：1源图像2输出检测矩形3特征向量和超平面的距离4移动步长(必须是block步长的整数倍)5边缘扩展6源图像图像每次缩小比例7聚类参数8聚类方式
// //   for(int i=0; i < found.size(); i++)  //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
// //   {  
// //       Rect r = found[i];  
// //       int j=0;  
// //       for(; j < found.size(); j++)  
// //           if(j != i && (r & found[j]) == r)  
// //               break;  
// //       if( j == found.size())  
// //           found_filtered.push_back(r);  
// //   }
//	//cout<<"找到的矩形框个数："<<found_filtered.size()<<endl;
//	//
//	////对检测到的图像进行分类
//	//for(int i=0; i<found_filtered.size(); i++)  
//	//{
//	//	cout<<"width:"<<found_filtered[i].width<<"  height:"<<found_filtered[i].height<<endl;//输出检测框图大小
//	//	vector<float> descriptors;//HOG描述子向量
//	//	Mat descriptorsMat = Mat::zeros(1, descriptorDim, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
//	//	Mat temp;
//	//	resize(src(found_filtered[i]),temp,WinSize);//调整检测结果图像尺寸
//
//	//	classifyHOG.compute(temp,descriptors);//计算HOG描述子
//	//	for(int i=0; i<descriptorDim; i++)  
//	//		descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
//
//	//	float classifyResult = classifySvm.predict(descriptorsMat);//进行输入框图类型预测
//
//	//	if (classifyResult == 1)//机器人
//	//	{
//	//		rectangle(dst, found_filtered[i], Scalar(255,0,0), 3);//在图中画出检测框
//	//		if (SAVESET)//是否保存检测数据
//	//		{
//	//			strstream ss;
//	//			string s;
//	//			ss<<ResultImageFile_1<<i<<".jpg";
//	//			ss>>s;
//	//			imwrite(s,src(found_filtered[i]));
//	//		}
//	//	} 
//	//	else if (classifyResult == 2)//障碍物
//	//	{
//	//		rectangle(dst, found_filtered[i], Scalar(0,255,0), 3);//在图中画出检测框
//	//		if (SAVESET)//是否保存检测数据
//	//		{
//	//			strstream ss;
//	//			string s;
//	//			ss<<ResultImageFile_2<<i<<".jpg";
//	//			ss>>s;
//	//			imwrite(s,src(found_filtered[i]));
//	//		}
//	//	}
//	//	else if (classifyResult ==3)//背景
//	//	{
//	//		rectangle(dst, found_filtered[i], Scalar(0,0,255), 3);//在图中画出检测框
//	//		if (SAVESET)//是否保存检测数据
//	//		{
//	//			strstream ss;
//	//			string s;
//	//			ss<<ResultImageFile_3<<i<<".jpg";
//	//			ss>>s;
//	//			imwrite(s,src(found_filtered[i]));
//	//		}
//	//	}
//	//	else//其他
//	//	{
//	//		rectangle(dst, found_filtered[i], Scalar(255,255,255), 3);//在图中画出检测框
//	//	}
//
//	//}
//
//	////储存检测图像结果
// //   imwrite(ResultImage,dst);  
// //   namedWindow("dst");  
// //   imshow("dst",dst);  
// //   waitKey(0);//注意：imshow之后必须加waitKey，否则无法显示图像  
//
//	////---------------------------------end---------------------------------------
//
//
//
//	//-------------------------读入视频进行机器人检测-----------------------------------
//	//变量定义
//	VideoCapture myVideo(TestVideo);//读取视频  
//	Mat src,dst;					//原始图像，处理后图像
//	
//	const char * sd1 = {"md "ResultVideoFile_1};//创建存放检测框图的文件夹
//	system(sd1);
//	const char * sd2 = {"md "ResultVideoFile_2};//创建存放检测框图的文件夹
//	system(sd2);
//	const char * sd3 = {"md "ResultVideoFile_3};//创建存放检测框图的文件夹
//	system(sd3);
//
//	//打开视频
//	if(!myVideo.isOpened()){cout<<"视频读取错误"<<endl;system("puase");return -1;}
//
//	//设置生成的视频
//	double videoRate=myVideo.get(CV_CAP_PROP_FPS);//获取帧率
//	int videoWidth=myVideo.get(CV_CAP_PROP_FRAME_WIDTH);//获取视频图像宽度
//	int videoHight=myVideo.get(CV_CAP_PROP_FRAME_HEIGHT);//获取视频图像高度
//	int videoDelay=1000/videoRate;//每帧之间的延迟与视频的帧率相对应（设置跑程序的时候播放视频的速率）
//	VideoWriter outputVideo(ResultVideo, CV_FOURCC('M', 'J', 'P', 'G'), videoRate, Size(videoWidth, videoHight));//设置视频类
//
//	//开始视频处理
//	bool stop = false;
//	for (int fnum = 1;!stop;fnum++)
//	{
//		//变量定义
//		if (!myVideo.read(src)){cout<<"视频结束"<<endl;waitKey(0); break;}//获取视频帧
//		//resize(videoFrame,videoFrame,Size(0,0),2,2);//调整视频图像的大小
//		src.copyTo(dst);
//		vector<Rect> found, found_filtered;//检测框容器
//
//		//对图片进行多尺度机器人检测 
//		cout<<"进行多尺度检测"<<endl;
//		detectHOG.detectMultiScale(src, found, HitThreshold, WinStride, Size(0,0), DetScale, 2, false);
//		//参数：1源图像2输出检测矩形3特征向量和超平面的距离4移动步长(必须是block步长的整数倍)5边缘扩展6源图像图像每次缩小比例7聚类参数8聚类方式
//
//		found_filtered = found;//不需要处理嵌套矩形框 则直接用检测的结果进行后续操作
//		//for(int i=0; i < found.size(); i++)  //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
//		//{  
//		//	Rect r = found[i];  
//		//	int j=0;  
//		//	for(; j < found.size(); j++)  
//		//		if(j != i && (r & found[j]) == r)  
//		//			break;  
//		//	if( j == found.size())  
//		//		found_filtered.push_back(r);  
//		//}
//		//cout<<"找到的矩形框个数："<<found_filtered.size()<<endl;
//		
//		//对检测到的图像进行分类
//		for(int i=0; i<found_filtered.size(); i++)  
//		{
//			cout<<"width:"<<found_filtered[i].width<<"  height:"<<found_filtered[i].height<<endl;//输出检测框图大小
//			vector<float> descriptors;//HOG描述子向量
//			Mat descriptorsMat = Mat::zeros(1, descriptorDim, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
//			Mat temp;
//			resize(src(found_filtered[i]),temp,WinSize);//调整检测结果图像尺寸
//
//			classifyHOG.compute(temp,descriptors);//计算HOG描述子
//			for(int i=0; i<descriptorDim; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
//
//			float classifyResult = classifySvm.predict(descriptorsMat);//进行输入框图类型预测
//
//			if (classifyResult == 1)//机器人
//			{
//				rectangle(dst, found_filtered[i], Scalar(255,0,0), 3);//在图中画出检测框
//				if (SAVESET)//是否保存检测数据
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_1<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found_filtered[i]));
//				}
//			} 
//			else if (classifyResult == 2)//障碍物
//			{
//				rectangle(dst, found_filtered[i], Scalar(0,255,0), 3);//在图中画出检测框
//				if (SAVESET)//是否保存检测数据
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_2<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found_filtered[i]));
//				}
//			}
//			else if (classifyResult ==3)//背景
//			{
//				rectangle(dst, found_filtered[i], Scalar(0,0,255), 3);//在图中画出检测框
//				if (SAVESET)//是否保存检测数据
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_3<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found_filtered[i]));
//				}
//			}
//			else//其他
//			{
//				rectangle(dst, found_filtered[i], Scalar(255,255,255), 3);//在图中画出检测框
//			}
//
//		}
//
//		//储存视频图像
//		outputVideo<<dst;
//		imshow("dst",dst);
//		if(waitKey(1)>=0)stop = true;//通过按键停止视频
//	}
//	//---------------------------------end---------------------------------------
//}  
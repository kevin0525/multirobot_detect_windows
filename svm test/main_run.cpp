//#include <iostream>  
//#include <fstream>  
//#include <strstream>
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>  
//#include <opencv2/imgproc/imgproc.hpp>  
//#include <opencv2/objdetect/objdetect.hpp>  
//#include <opencv2/ml/ml.hpp>  
//#include "someMethod.h"
//#include "parameter.h"
//using namespace std;  
//using namespace cv;  
//
////-----------------------主函数----------------------------
////---------------------------------------------------------
//
//int main()  
//{
//	//变量定义
//    int descriptorDimDetect;//HOG描述子的维数：[(检测窗口长-block长)/block步长+1]*[(检测窗口高-block高)/block步高+1]*bin个数*(block长/cell长)*(block高/cell高)
//	int descriptorDimClassify;
//    MySVM detectSvm;//检测SVM
//	MySVM classifySvm;//分类SVM
//	detectSvm.load(DetectSvmName);
//	classifySvm.load(ClassifySvmName);
//
//	//----------------进行机器人和障碍物的检测与分类-----------------
//	//---------------------------------------------------------------
//	//变量定义
//	HOGDescriptor detectHOG(WinSizeDetect,BlockSizeDetect,BlockStrideDetect,CellSizeDetect,NbinsDetect);//分类HOG检测器
//	HOGDescriptor classifyHOG(WinSizeClassify,BlockSizeClassify,BlockStrideClassify,CellSizeClassify,NbinsClassify);//检测HOG：用于计算检测结果的HOG特征向量
//    descriptorDimDetect = detectSvm.get_var_count();//特征向量的维数，即HOG描述子的维数（和前面训练时的大小一样，添加此句是为了在不训练时也能拿到维数）
//    descriptorDimClassify = classifySvm.get_var_count();//特征向量的维数，即HOG描述子的维数（和前面训练时的大小一样，添加此句是为了在不训练时也能拿到维数）
//	int supportVectorDetectNum = detectSvm.get_support_vector_count();//支持向量的个数
//	int supportVectorCassifyNum = classifySvm.get_support_vector_count();//支持向量的个数
//    cout<<"Detect支持向量个数："<<supportVectorDetectNum<<endl;  
//	cout<<"Cassify支持向量个数："<<supportVectorCassifyNum<<endl;  
//    Mat alphaDetectMat = Mat::zeros(1, supportVectorDetectNum, CV_32FC1);//alpha向量，长度等于支持向量个数
//	Mat alphaClassifyMat = Mat::zeros(1, supportVectorCassifyNum, CV_32FC1);//alpha向量，长度等于支持向量个数 
//    Mat supportVectorDetectMat = Mat::zeros(supportVectorDetectNum, descriptorDimDetect, CV_32FC1);//支持向量矩阵  
//	Mat supportVectorClassifyMat = Mat::zeros(supportVectorCassifyNum, descriptorDimClassify, CV_32FC1);//支持向量矩阵  
//    Mat resultDetectMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//alpha向量乘以支持向量矩阵的结果  
//	Mat resultClassifyMat = Mat::zeros(1, descriptorDimClassify, CV_32FC1);//alpha向量乘以支持向量矩阵的结果  
//
//    //计算w矩阵
//    for(int i=0; i<supportVectorDetectNum; i++)//将支持向量的数据复制到supportVectorMat矩阵中  
//	{
//        const float * pSVData = detectSvm.get_support_vector(i);//返回第i个支持向量的数据指针  
//        for(int j=0; j<descriptorDimDetect; j++)  
//            supportVectorDetectMat.at<float>(i,j) = pSVData[j];  
//    }
//	for(int i=0; i<supportVectorCassifyNum; i++)//将支持向量的数据复制到supportVectorMat矩阵中  
//	{
//		const float * pSVData = classifySvm.get_support_vector(i);//返回第i个支持向量的数据指针  
//		for(int j=0; j<descriptorDimClassify; j++)  
//			supportVectorClassifyMat.at<float>(i,j) = pSVData[j];  
//	}
//    double * pAlphaDetectData = detectSvm.get_alpha_vector();//返回SVM的决策函数中的alpha向量  
//	double * pAlphaClassifyData = classifySvm.get_alpha_vector();//返回SVM的决策函数中的alpha向量  
//    for(int i=0; i<supportVectorDetectNum; i++)//将alpha向量的数据复制到alphaMat中  
//        alphaDetectMat.at<float>(0,i) = pAlphaDetectData[i];  
//	for(int i=0; i<supportVectorCassifyNum; i++)//将alpha向量的数据复制到alphaMat中  
//		alphaClassifyMat.at<float>(0,i) = pAlphaClassifyData[i];  
//	resultDetectMat = -1 * alphaDetectMat * supportVectorDetectMat;//计算-(alphaMat * supportVectorMat),结果放到resultMat中
//    resultClassifyMat = -1 * alphaClassifyMat * supportVectorClassifyMat;//计算-(alphaMat * supportVectorMat),结果放到resultMat中 
//
//    //得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子  
//    vector<float> myDetector;//基于Hog特征的SVM检测子（w+b）
//    for(int i=0; i<descriptorDimDetect; i++)//将resultMat中的数据复制到数组myDetector中  
//        myDetector.push_back(resultDetectMat.at<float>(0,i));  
//    myDetector.push_back(detectSvm.get_rho());//最后添加偏移量rho，得到检测子  
//    cout<<"基于Hog特征的SVM检测子维数(w+b)："<<myDetector.size()<<endl;
//
//	//设置SVMDetector检测子
//    detectHOG.setSVMDetector(myDetector);  
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
//	//label
//	LabelRobot label_irobot(50,10);
//	LabelRobot label_obstacle(50,4);
//
//	//开始视频处理
//	bool stop = false;
//	for (int fnum = 1;!stop;fnum++)
//	{
//		cout<<fnum<<endl;
//		//变量定义
//		if (!myVideo.read(src)){cout<<"视频结束"<<endl;waitKey(0); break;}//获取视频帧
//		//resize(videoFrame,videoFrame,Size(0,0),2,2);//调整视频图像的大小
//		src.copyTo(dst);
//		vector<Rect> found;//检测框容器
//
//		//对图片进行多尺度机器人检测 
//		cout<<"进行多尺度检测"<<endl;
//		detectHOG.detectMultiScale(src, found, HitThreshold, WinStride, Size(0,0), DetScale, 0.2, true);
//		//参数：1源图像2输出检测矩形3特征向量和超平面的距离4移动步长(必须是block步长的整数倍)5边缘扩展6源图像图像每次缩小比例7聚类参数8聚类方式
//
//		//label
//		vector<RobotMessage> irobots_message;
//		vector<RobotMessage> obstacles_message;
//
//		//对检测到的图像进行分类
//		for(int i=0; i<found.size(); i++)  
//		{
//			cout<<"width:"<<found[i].width<<"  height:"<<found[i].height<<endl;//输出检测框图大小
//			vector<float> descriptors;//HOG描述子向量
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassify, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
//			Mat temp;
//			resize(src(found[i]),temp,WinSizeClassify);//调整检测结果图像尺寸
//
//			classifyHOG.compute(temp,descriptors);//计算HOG描述子
//			for(int i=0; i<descriptorDimClassify; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
//
//			float classifyResult = classifySvm.predict(descriptorsMat);//进行输入框图类型预测
//
//			if (classifyResult == 1)//机器人
//			{
//				irobots_message.insert(irobots_message.end(),RobotMessage(found[i]));
//				rectangle(dst, found[i], Scalar(255,0,0), 3);//在图中画出检测框
//				if (SAVESET)//是否保存检测数据
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_1<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found[i]));
//				}
//			} 
//			else if (classifyResult == 2)//障碍物
//			{
//				obstacles_message.insert(obstacles_message.end(),RobotMessage(found[i]));
//				rectangle(dst, found[i], Scalar(0,255,0), 3);//在图中画出检测框
//				if (SAVESET)//是否保存检测数据
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_2<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found[i]));
//				}
//			}
//			else if (classifyResult ==3)//背景
//			{
//				rectangle(dst, found[i], Scalar(0,0,255), 3);//在图中画出检测框
//				if (SAVESET)//是否保存检测数据
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_3<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found[i]));
//				}
//			}
//		}
//
//		//label
//		label_irobot.input(irobots_message);
//		label_irobot.getLabel(irobots_message);
//		label_obstacle.input(obstacles_message);
//		label_obstacle.getLabel(obstacles_message);
//
//		for (int i = 0; i < irobots_message.size(); i++)
//		{
//			char label[8];
//			sprintf(label,"%d",irobots_message[i].label);
//			//在图像上放置坐标 参数为：图片，文字，文字在图像中的左下角坐标，字体类型，大小，颜色，粗细，线型
//			putText(dst,label,irobots_message[i].center,FONT_HERSHEY_SIMPLEX,2,CV_RGB(0,0,0),2,10);
//		}
//		for (int i = 0; i < obstacles_message.size(); i++)
//		{
//			char label[8];
//			sprintf(label,"%d",obstacles_message[i].label);
//			//在图像上放置坐标 参数为：图片，文字，文字在图像中的左下角坐标，字体类型，大小，颜色，粗细，线型
//			putText(dst,label,obstacles_message[i].center,FONT_HERSHEY_SIMPLEX,2,CV_RGB(0,0,0),2,10);
//		}
//
//		//储存视频图像
//		outputVideo<<dst;
//		imshow("dst",dst);
//		if(waitKey(1)>=0)stop = true;//通过按键停止视频
//	}
//	//---------------------------------end---------------------------------------
//}  
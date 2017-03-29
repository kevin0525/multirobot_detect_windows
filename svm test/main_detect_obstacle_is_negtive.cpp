#include <iostream>  
#include <fstream>  
#include <strstream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  
#include "someMethod.h"
#include "parameter.h"
using namespace std;  
using namespace cv;  


//-----------------------主函数----------------------------
//---------------------------------------------------------

int main()  
{
	//变量定义
	HOGDescriptor detectHOG(WinSizeDetect,BlockSizeDetect,BlockStrideDetect,CellSizeDetect,NbinsDetect);//HOG描述子：检测窗口，block尺寸，block步长，cell尺寸，直方图bin个数 
	int descriptorDimDetect;//HOG描述子的维数：[(检测窗口长-block长)/block步长+1]*[(检测窗口高-block高)/block步高+1]*bin个数*(block长/cell长)*(block高/cell高)
	MySVM detectSvm;//检测SVM

	//文件变量定义
	string ImgName;//图片名
	ifstream IrobotName((string)IrobotSetFile+SetName);//机器人样本图片的文件名列表
	ifstream ObstacleName((string)ObstacleSetFile+SetName);//障碍样本图片的文件名列表
	ifstream BackgroundName((string)BackgroundSetFile+SetName);//背景样本图片的文件名列表
	ifstream HardBackgroundName((string)HardBackgroundSetFile+SetName);//Hard背景样本图片的文件名列表 

	int irobTypeArray[IrobotSetNo];//初始化样本名称顺序数组
	int obstTypeArray[ObstacleSetNo];
	int backTypeArray[BackgroundSetNo];
	int hardTypeArray[HardBackgroundSetNo];
	random(irobTypeArray, IrobotSetNo);//打乱样本名称顺序数组
	random(obstTypeArray, ObstacleSetNo);
	random(backTypeArray, BackgroundSetNo);
	random(hardTypeArray, HardBackgroundSetNo);
	typeHandle(irobTypeArray,IrobotSetNo,IrobotTrainNo,IrobotVaildNo);//样本类型赋值（train,vaild,test）
	typeHandle(obstTypeArray,ObstacleSetNo,ObstacleTrainNo,ObstacleVaildNo);
	typeHandle(backTypeArray,BackgroundSetNo,BackgroundTrainNo,BackgroundVaildNo);
	typeHandle(hardTypeArray,HardBackgroundSetNo,HardBackgroundTrainNo,HardBackgroundVaildNo);


	//----------------训练分类器or直接读取分类器---------------------
	//---------------------------------------------------------------
	if(TRAIN) //训练分类器，并保存XML文件
	{
		//训练变量定义
		Mat sampleFeatureDetectMat;//训练SVM的特征向量矩阵：行数=样本个数，列数=特征向量维数
		Mat detectLabelMat;//检测SVM的的类别向量：行数=样本个数，列数=1：1表示有机器人或障碍，-1表示无机器人和障碍

		//1.处理机器人样本图片
		for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName,ImgName); setIndex++)  
		{
			if (irobTypeArray[setIndex]==0)
			{
				ImgName = IrobotSetFile + ImgName;//加上机器人样本的路径名  
				cout<<"处理："<<ImgName<<endl;  
				Mat src = imread(ImgName);//读取图片
				Mat srcDetect;
				resize(src,srcDetect,WinSizeDetect);//将训练样本归一化为检测窗口的大小
				if(SHOWSET)//是否显示训练样本
				{
					imshow("【机器人样本】",src);
					if (waitKey(1)>0){return 0;}//通过按键中断程序
				}

				vector<float> descriptorsDetect;//HOG描述子向量
				detectHOG.compute(srcDetect,descriptorsDetect);//计算HOG描述子

				if(num == 0)//处理第一个样本时初始化特征向量矩阵和类别矩阵  
				{
					descriptorDimDetect = descriptorsDetect.size();//HOG描述子的维数
					sampleFeatureDetectMat = Mat::zeros(AllTrainNo, descriptorDimDetect, CV_32FC1);
					detectLabelMat = Mat::zeros(AllTrainNo, 1, CV_32FC1);  
				} 

				//将计算好的HOG描述子复制到特征向量矩阵和类别矩阵
				for(int i=0; i<descriptorDimDetect; i++)
					sampleFeatureDetectMat.at<float>(num,i) = descriptorsDetect[i];//第num个样本的特征向量中的第i个元素  
				detectLabelMat.at<float>(num,0) = 1;//1表示有机器人
				num++;
			}
		}  

		//2.处理障碍样本图片
		for(int setIndex=0,num=IrobotTrainNo; setIndex<ObstacleSetNo && getline(ObstacleName,ImgName); setIndex++)  
		{
			if (obstTypeArray[setIndex]==0)
			{
				ImgName = ObstacleSetFile + ImgName;//加上障碍样本的路径名
				cout<<"处理："<<ImgName<<endl;  
				Mat src = imread(ImgName);//读取图片
				Mat srcDetect;
				resize(src,srcDetect,WinSizeDetect);//将训练样本归一化为检测窗口的大小
				if(SHOWSET)//是否显示训练样本
				{
					imshow("【机器人样本】",src);
					if (waitKey(1)>0){return 0;}//通过按键中断程序
				}

				vector<float> descriptorsDetect;//HOG描述子向量
				detectHOG.compute(srcDetect,descriptorsDetect);//计算HOG描述子

				//将计算好的HOG描述子复制到特征向量矩阵和类别矩阵
				for(int i=0; i<descriptorDimDetect; i++)
					sampleFeatureDetectMat.at<float>(num,i) = descriptorsDetect[i];//第num个样本的特征向量中的第i个元素  
				detectLabelMat.at<float>(num,0) = -1;//1表示有机器人
				num++;
			}
		}

		//3.处理背景样本图片
		for(int setIndex=0,num=IrobotTrainNo+ObstacleTrainNo; setIndex<BackgroundSetNo && getline(BackgroundName,ImgName); setIndex++)  
		{
			if (backTypeArray[setIndex]==0)
			{
				ImgName = BackgroundSetFile + ImgName;//加上背景样本的路径名  
				cout<<"处理："<<ImgName<<endl;  
				Mat src = imread(ImgName);//读取图片
				Mat srcDetect;
				resize(src,srcDetect,WinSizeDetect);//将训练样本归一化为检测窗口的大小
				if(SHOWSET)//是否显示训练样本
				{
					imshow("【机器人样本】",src);
					if (waitKey(1)>0){return 0;}//通过按键中断程序
				}

				vector<float> descriptorsDetect;//HOG描述子向量
				detectHOG.compute(srcDetect,descriptorsDetect);//计算HOG描述子

				//将计算好的HOG描述子复制到特征向量矩阵和类别矩阵
				for(int i=0; i<descriptorDimDetect; i++)
					sampleFeatureDetectMat.at<float>(num,i) = descriptorsDetect[i];//第num个样本的特征向量中的第i个元素  
				detectLabelMat.at<float>(num,0) = -1;//1表示有机器人
				num++;
			}
		}

		//4.处理Hard背景样本图片
		for(int setIndex=0,num=IrobotTrainNo+ObstacleTrainNo+BackgroundTrainNo; setIndex<HardBackgroundSetNo && getline(HardBackgroundName,ImgName); setIndex++)  
		{
			if (hardTypeArray[setIndex]==0)
			{
				ImgName = HardBackgroundSetFile + ImgName;//加上Hard背景样本的路径名
				cout<<"处理："<<ImgName<<endl;
				Mat src = imread(ImgName);//读取图片
				Mat srcDetect;
				resize(src,srcDetect,WinSizeDetect);//将训练样本归一化为检测窗口的大小
				if(SHOWSET)//是否显示训练样本
				{
					imshow("【机器人样本】",src);
					if (waitKey(1)>0){return 0;}//通过按键中断程序
				}

				vector<float> descriptorsDetect;//HOG描述子向量
				detectHOG.compute(srcDetect,descriptorsDetect);//计算HOG描述子

				//将计算好的HOG描述子复制到特征向量矩阵和类别矩阵
				for(int i=0; i<descriptorDimDetect; i++)
					sampleFeatureDetectMat.at<float>(num,i) = descriptorsDetect[i];//第num个样本的特征向量中的第i个元素  
				detectLabelMat.at<float>(num,0) = -1;//1表示有机器
				num++;
			}
		}  

		//5.训练检测SVM分类器  
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria detectCriteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);  
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams detectParam(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, detectCriteria);  
		cout<<"开始训练检测SVM分类器"<<endl;  
		detectSvm.train(sampleFeatureDetectMat, detectLabelMat, Mat(), Mat(), detectParam);//训练分类器
		cout<<"训练完成"<<endl;  
		detectSvm.save(DetectSvmName);//将训练好的SVM模型保存为xml文件  
	}  
	else //若TRAIN为false，从XML文件读取训练好的分类器  
	{  
		detectSvm.load(DetectSvmName);
	}  

	//----------------进行机器人和障碍物的检测与分类-----------------
	//---------------------------------------------------------------
	//变量定义
	descriptorDimDetect = detectSvm.get_var_count();//特征向量的维数，即HOG描述子的维数（和前面训练时的大小一样，添加此句是为了在不训练时也能拿到维数）
	int supportVectorNumDetect = detectSvm.get_support_vector_count();//支持向量的个数
	cout<<"Detect支持向量个数："<<supportVectorNumDetect<<endl;  
	Mat alphaDetectMat = Mat::zeros(1, supportVectorNumDetect, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorDetectMat = Mat::zeros(supportVectorNumDetect, descriptorDimDetect, CV_32FC1);//支持向量矩阵 
	Mat resultDetectMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//alpha向量乘以支持向量矩阵的结果 

	//计算w矩阵
	for(int i=0; i<supportVectorNumDetect; i++)//将支持向量的数据复制到supportVectorMat矩阵中  
	{
		const float * pSVData = detectSvm.get_support_vector(i);//返回第i个支持向量的数据指针  
		for(int j=0; j<descriptorDimDetect; j++)  
			supportVectorDetectMat.at<float>(i,j) = pSVData[j];  
	}

	double * pAlphaDataDetect = detectSvm.get_alpha_vector();//返回SVM的决策函数中的alpha向量  
	for(int i=0; i<supportVectorNumDetect; i++)//将alpha向量的数据复制到alphaMat中  
		alphaDetectMat.at<float>(0,i) = pAlphaDataDetect[i];  
	resultDetectMat = -1 * alphaDetectMat * supportVectorDetectMat;//计算-(alphaMat * supportVectorMat),结果放到resultMat中 

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子  
	vector<float> myDetector;//基于Hog特征的SVM检测子（w+b）
	for(int i=0; i<descriptorDimDetect; i++)//将resultMat中的数据复制到数组myDetector中  
		myDetector.push_back(resultDetectMat.at<float>(0,i));  
	myDetector.push_back(detectSvm.get_rho());//最后添加偏移量rho，得到检测子  
	cout<<"基于Hog特征的SVM检测子维数(w+b)："<<myDetector.size()<<endl;

	//设置SVMDetector检测子
	detectHOG.setSVMDetector(myDetector);  

	//-----------------------------读入train样本检验准确率-------------------------
	//0.变量定义
	ifstream IrobotName2((string)IrobotSetFile+SetName);//机器人样本图片的文件名列表
	ifstream ObstacleName2((string)ObstacleSetFile+SetName);//障碍样本图片的文件名列表
	ifstream BackgroundName2((string)BackgroundSetFile+SetName);//背景样本图片的文件名列表
	ifstream HardBackgroundName2((string)HardBackgroundSetFile+SetName);//Hard背景样本图片的文件名列表 
	float TPtrain = 0,//TP:实际为真，预测为真
		TNtrain = 0,//TN:实际为假，预测为假
		FPtrain = 0,//FP:实际为假，预测为真
		FNtrain = 0;//FN:实际为真，预测为假
	float PrecisionTrain = -1,//Precision = TP/(TP+FP);
		RecallTrain = -1,//Recall = TP/(TP+FN);
		F1ScoreTrain = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)

	//计算检测train正样本准确率
	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName2,ImgName); setIndex++)  
	{
		if (irobTypeArray[setIndex]==0)
		{
			ImgName = IrobotSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				TPtrain = TPtrain + 1;
			} 
			else
			{
				FNtrain = FNtrain + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName2,ImgName); setIndex++)  
	{
		if (obstTypeArray[setIndex]==0)
		{
			ImgName = ObstacleSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPtrain = FPtrain + 1;
			} 
			else
			{
				TNtrain = TNtrain + 1;
			}
		}
	}
	//计算检测train负样本准确率
	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName2,ImgName); setIndex++)  
	{
		if (backTypeArray[setIndex]==0)
		{
			ImgName = BackgroundSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPtrain = FPtrain + 1;
			} 
			else
			{
				TNtrain = TNtrain + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName2,ImgName); setIndex++)  
	{
		if (hardTypeArray[setIndex]==0)
		{
			ImgName = HardBackgroundSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPtrain = FPtrain + 1;
			} 
			else
			{
				TNtrain = TNtrain + 1;
			}
		}
	}
	//---------------------------------------end----------------------------------



	//-----------------------------读入vaild样本检验准确率-------------------------
	//0.变量定义
	ifstream IrobotName3((string)IrobotSetFile+SetName);//机器人样本图片的文件名列表
	ifstream ObstacleName3((string)ObstacleSetFile+SetName);//障碍样本图片的文件名列表
	ifstream BackgroundName3((string)BackgroundSetFile+SetName);//背景样本图片的文件名列表
	ifstream HardBackgroundName3((string)HardBackgroundSetFile+SetName);//Hard背景样本图片的文件名列表 
	float TPvaild = 0,//TP:实际为真，预测为真
		TNvaild = 0,//TN:实际为假，预测为假
		FPvaild = 0,//FP:实际为假，预测为真
		FNvaild = 0;//FN:实际为真，预测为假
	float PrecisionVaild = -1,//Precision = TP/(TP+FP);
		RecallVaild = -1,//Recall = TP/(TP+FN);
		F1ScoreVaild = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)

	//计算检测vaid正样本准确率
	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName3,ImgName); setIndex++)  
	{
		if (irobTypeArray[setIndex]==1)
		{
			ImgName = IrobotSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				TPvaild = TPvaild + 1;
			} 
			else
			{
				FNvaild = FNvaild + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName3,ImgName); setIndex++)  
	{
		if (obstTypeArray[setIndex]==1)
		{
			ImgName = ObstacleSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPvaild = FPvaild + 1;
			} 
			else
			{
				TNvaild = TNvaild + 1;
			}
		}
	}
	//计算检测vaild负样本准确率
	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName3,ImgName); setIndex++)  
	{
		if (backTypeArray[setIndex]==1)
		{
			ImgName = BackgroundSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPvaild = FPvaild + 1;
			} 
			else
			{
				TNvaild = TNvaild + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName3,ImgName); setIndex++)  
	{
		if (hardTypeArray[setIndex]==1)
		{
			ImgName = HardBackgroundSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPvaild = FPvaild + 1;
			} 
			else
			{
				TNvaild = TNvaild + 1;
			}
		}
	}
	//---------------------------------------end----------------------------------

	//-----------------------------读入test样本检验准确率-------------------------
	//0.变量定义
	ifstream IrobotName4((string)IrobotSetFile+SetName);//机器人样本图片的文件名列表
	ifstream ObstacleName4((string)ObstacleSetFile+SetName);//障碍样本图片的文件名列表
	ifstream BackgroundName4((string)BackgroundSetFile+SetName);//背景样本图片的文件名列表
	ifstream HardBackgroundName4((string)HardBackgroundSetFile+SetName);//Hard背景样本图片的文件名列表 
	float TPtest = 0,//TP:实际为真，预测为真
		TNtest = 0,//TN:实际为假，预测为假
		FPtest = 0,//FP:实际为假，预测为真
		FNtest = 0;//FN:实际为真，预测为假
	float PrecisionTest = -1,//Precision = TP/(TP+FP);
		RecallTest = -1,//Recall = TP/(TP+FN);
		F1ScoreTest = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)

	//计算检测vaid正样本准确率
	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName4,ImgName); setIndex++)  
	{
		if (irobTypeArray[setIndex]==2)
		{
			ImgName = IrobotSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				TPtest = TPtest + 1;
			} 
			else
			{
				FNtest = FNtest + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName4,ImgName); setIndex++)  
	{
		if (obstTypeArray[setIndex]==2)
		{
			ImgName = ObstacleSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPtest = FPtest + 1;
			} 
			else
			{
				TNtest = TNtest + 1;
			}
		}
	}
	//计算检测test负样本准确率
	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName4,ImgName); setIndex++)  
	{
		if (backTypeArray[setIndex]==2)
		{
			ImgName = BackgroundSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPtest = FPtest + 1;
			} 
			else
			{
				TNtest = TNtest + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName4,ImgName); setIndex++)  
	{
		if (hardTypeArray[setIndex]==2)
		{
			ImgName = HardBackgroundSetFile + ImgName;//加上机器人样本的路径名  
			cout<<"处理："<<ImgName<<endl;  
			Mat src = imread(ImgName);//读取图片  
			resize(src,src,WinSizeDetect);//将训练样本归一化为检测窗口的大小
			if(SHOWSET)//是否显示训练样本
			{
				imshow("【机器人样本】",src);
				if (waitKey(1)>0){return 0;}//通过按键中断程序
			}

			vector<float> descriptors;//HOG描述子向量
			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//分类用的HOG特征向量矩阵：行数=1，列数=特征向量维数
			detectHOG.compute(src,descriptors);//计算HOG描述子
			for(int i=0; i<descriptorDimDetect; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//特征向量矩阵赋值
			float detectResult = detectSvm.predict(descriptorsMat);//进行输入框图类型预测
			if (detectResult == 1)//计算结果
			{
				FPtest = FPtest + 1;
			} 
			else
			{
				TNtest = TNtest + 1;
			}
		}
	}
	//---------------------------------------end----------------------------------

	//---------------------------------指标计算-----------------------------------
	PrecisionTrain = TPtrain/(TPtrain+FPtrain);//Precision = TP/(TP+FP);
	RecallTrain = TPtrain/(TPtrain+FNtrain);//Recall = TP/(TP+FN);
	F1ScoreTrain = 2*PrecisionTrain*RecallTrain/(PrecisionTrain+RecallTrain);//F1Score = 2*Precision*Recall/(Precision+Recall);
	cout<<"PrecisionTrain:"<<PrecisionTrain<<endl;
	cout<<"RecallTrain:"<<RecallTrain<<endl;
	cout<<"F1ScoreTrain:"<<F1ScoreTrain<<endl;
	cout<<endl;

	PrecisionVaild = TPvaild/(TPvaild+FPvaild);//Precision = TP/(TP+FP);
	RecallVaild = TPvaild/(TPvaild+FNvaild);//Recall = TP/(TP+FN);
	F1ScoreVaild = 2*PrecisionVaild*RecallVaild/(PrecisionVaild+RecallVaild);//F1Score = 2*Precision*Recall/(Precision+Recall);
	cout<<"PrecisionVaild:"<<PrecisionVaild<<endl;
	cout<<"RecallVaild:"<<RecallVaild<<endl;
	cout<<"F1ScoreVaild:"<<F1ScoreVaild<<endl;
	cout<<endl;

	PrecisionTest = TPtest/(TPtest+FPtest);//Precision = TP/(TP+FP);
	RecallTest = TPtest/(TPtest+FNtest);//Recall = TP/(TP+FN);
	F1ScoreTest = 2*PrecisionTest*RecallTest/(PrecisionTest+RecallTest);//F1Score = 2*Precision*Recall/(Precision+Recall);
	cout<<"PrecisionTest:"<<PrecisionTest<<endl;
	cout<<"RecallTest:"<<RecallTest<<endl;
	cout<<"F1ScoreTest:"<<F1ScoreTest<<endl;
	//---------------------------------------end----------------------------------
	getchar();
	return 0;
}  
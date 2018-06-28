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


//-----------------------������----------------------------
//---------------------------------------------------------

int main()  
{
	//��������
	HOGDescriptor classifyHOG(WinSizeClassify,BlockSizeClassify,BlockStrideClassify,CellSizeClassify,NbinsClassify);
	int descriptorDimClassfy;
	MySVM classifySvm;//����SVM

	//�ļ���������
	string ImgName;//ͼƬ��
	ifstream IrobotName((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
	ifstream ObstacleName((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
	ifstream BackgroundName((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
	ifstream HardBackgroundName((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 

	int irobTypeArray[IrobotSetNo];//��ʼ����������˳������
	int obstTypeArray[ObstacleSetNo];
	int backTypeArray[BackgroundSetNo];
	int hardTypeArray[HardBackgroundSetNo];
	random(irobTypeArray, IrobotSetNo);//������������˳������
	random(obstTypeArray, ObstacleSetNo);
	random(backTypeArray, BackgroundSetNo);
	random(hardTypeArray, HardBackgroundSetNo);
	typeHandle(irobTypeArray,IrobotSetNo,IrobotTrainNo,IrobotVaildNo);//�������͸�ֵ��train,vaild,test��
	typeHandle(obstTypeArray,ObstacleSetNo,ObstacleTrainNo,ObstacleVaildNo);
	typeHandle(backTypeArray,BackgroundSetNo,BackgroundTrainNo,BackgroundVaildNo);
	typeHandle(hardTypeArray,HardBackgroundSetNo,HardBackgroundTrainNo,HardBackgroundVaildNo);


	//----------------ѵ��������orֱ�Ӷ�ȡ������---------------------
	//---------------------------------------------------------------
	if(TRAIN) //ѵ����������������XML�ļ�
	{
		//ѵ����������
		Mat sampleFeatureClassifyMat;//ѵ��SVM������������������=��������������=��������ά��
		Mat classifyLabelMat;//����SVM�ĵ��������������=��������������=1��1��ʾ�л����ˣ�2��ʾ���ϰ���3��ʾ�б���
		//1.�������������ͼƬ
		for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName,ImgName); setIndex++)  
		{
			if (irobTypeArray[setIndex]==0)
			{
				ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
				cout<<"����"<<ImgName<<endl;  
				Mat src = imread(ImgName);//��ȡͼƬ
				Mat srcClassify;
				resize(src,srcClassify,WinSizeClassify);
				if(SHOWSET)//�Ƿ���ʾѵ������
				{
					imshow("��������������",src);
					if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
				}

				vector<float> descriptorsClassify;//HOG����������
				classifyHOG.compute(srcClassify,descriptorsClassify);

				if(num == 0)//�����һ������ʱ��ʼ���������������������  
				{
					descriptorDimClassfy = descriptorsClassify.size();//HOG�����ӵ�ά��
					sampleFeatureClassifyMat = Mat::zeros(AllTrainNo, descriptorDimClassfy, CV_32FC1); 
					classifyLabelMat = Mat::zeros(AllTrainNo, 1, CV_32FC1);  
				} 

				//������õ�HOG�����Ӹ��Ƶ�������������������� 
				for(int i=0; i<descriptorDimClassfy; i++)
					sampleFeatureClassifyMat.at<float>(num,i) = descriptorsClassify[i];//��num�����������������еĵ�i��Ԫ��
				classifyLabelMat.at<float>(num,0) = 1;//1��ʾ�л�����
				num++;
			}
		}  

		//2.�����ϰ�����ͼƬ
		for(int setIndex=0,num=IrobotTrainNo; setIndex<ObstacleSetNo && getline(ObstacleName,ImgName); setIndex++)  
		{
			if (obstTypeArray[setIndex]==0)
			{
				ImgName = ObstacleSetFile + ImgName;//�����ϰ�������·����
				cout<<"����"<<ImgName<<endl;  
				Mat src = imread(ImgName);//��ȡͼƬ
				Mat srcClassify;
				
				resize(src,srcClassify,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
				if(SHOWSET)//�Ƿ���ʾѵ������
				{
					imshow("��������������",src);
					if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
				}

				vector<float> descriptorsClassify;//HOG����������
				classifyHOG.compute(srcClassify,descriptorsClassify);//����HOG������

				//������õ�HOG�����Ӹ��Ƶ��������������������
				for(int i=0; i<descriptorDimClassfy; i++)
					sampleFeatureClassifyMat.at<float>(num,i) = descriptorsClassify[i];//��num�����������������еĵ�i��Ԫ��
				classifyLabelMat.at<float>(num,0) = 2;//2��ʾ���ϰ�
				num++;
			}
		}

		//3.����������ͼƬ
		for(int setIndex=0,num=IrobotTrainNo+ObstacleTrainNo; setIndex<BackgroundSetNo && getline(BackgroundName,ImgName); setIndex++)  
		{
			if (backTypeArray[setIndex]==0)
			{
				ImgName = BackgroundSetFile + ImgName;//���ϱ���������·����  
				cout<<"����"<<ImgName<<endl;  
				Mat src = imread(ImgName);//��ȡͼƬ
				Mat srcClassify;
				resize(src,srcClassify,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
				if(SHOWSET)//�Ƿ���ʾѵ������
				{
					imshow("��������������",src);
					if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
				}

				vector<float> descriptorsClassify;//HOG����������
				classifyHOG.compute(srcClassify,descriptorsClassify);//����HOG������

				//������õ�HOG�����Ӹ��Ƶ��������������������
				for(int i=0; i<descriptorDimClassfy; i++)
					sampleFeatureClassifyMat.at<float>(num,i) = descriptorsClassify[i];//��num�����������������еĵ�i��Ԫ��  
				classifyLabelMat.at<float>(num,0) = 3;//3��ʾ����
				num++;
			}
		}

		//4.����Hard��������ͼƬ
		for(int setIndex=0,num=IrobotTrainNo+ObstacleTrainNo+BackgroundTrainNo; setIndex<HardBackgroundSetNo && getline(HardBackgroundName,ImgName); setIndex++)  
		{
			if (hardTypeArray[setIndex]==0)
			{
				ImgName = HardBackgroundSetFile + ImgName;//����Hard����������·����
				cout<<"����"<<ImgName<<endl;
				Mat src = imread(ImgName);//��ȡͼƬ
				Mat srcClassify;
				resize(src,srcClassify,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
				if(SHOWSET)//�Ƿ���ʾѵ������
				{
					imshow("��������������",src);
					if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
				}

				vector<float> descriptorsClassify;//HOG����������
				classifyHOG.compute(srcClassify,descriptorsClassify);//����HOG������

				//������õ�HOG�����Ӹ��Ƶ��������������������
				for(int i=0; i<descriptorDimClassfy; i++)
					sampleFeatureClassifyMat.at<float>(num,i) = descriptorsClassify[i];//��num�����������������еĵ�i��Ԫ�� 
				classifyLabelMat.at<float>(num,0) = 3;//3��ʾ����
				num++;
			}
		}  

		//6.ѵ������SVM������  
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria classifyCriteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10000, FLT_EPSILON);  
		//SVM������SVM����ΪC_SVC��RBF�˺���,gammaΪ1
		CvSVMParams classifyParam(CvSVM::C_SVC, CvSVM::RBF, 0, 0.7, 0, 1, 0, 0, 0, classifyCriteria);  
		cout<<"��ʼѵ������SVM������"<<endl;  
		classifySvm.train(sampleFeatureClassifyMat, classifyLabelMat, Mat(), Mat(), classifyParam);//ѵ��������
		cout<<"ѵ�����"<<endl;  
		classifySvm.save(ClassifySvmName);//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�  
	}  
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����  
	{  
		classifySvm.load(ClassifySvmName);
	}  

	//----------------���л����˺��ϰ���ļ�������-----------------
	//---------------------------------------------------------------
	//��������
	descriptorDimClassfy = classifySvm.get_var_count();
	int supportVectorNumClassify = classifySvm.get_support_vector_count(); 
	cout<<"Classify֧������������"<<supportVectorNumClassify<<endl; 
	Mat alphaClassifyMat = Mat::zeros(1, supportVectorNumClassify, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorClassifyMat = Mat::zeros(supportVectorNumClassify, descriptorDimClassfy, CV_32FC1);
	Mat resultClassifyMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//alpha��������֧����������Ľ�� 

	//����w����
	for(int i=0; i<supportVectorNumClassify; i++)//��֧�����������ݸ��Ƶ�supportVectorMat������
	{
		const float * pSVData = classifySvm.get_support_vector(i);//���ص�i��֧������������ָ��
		for(int j=0; j<descriptorDimClassfy; j++)  
			supportVectorClassifyMat.at<float>(i,j) = pSVData[j];  
	}
	double * pAlphaDataClassify = classifySvm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
	for(int i=0; i<supportVectorNumClassify; i++)//��alpha���������ݸ��Ƶ�alphaMat��
		alphaClassifyMat.at<float>(0,i) = pAlphaDataClassify[i];  
	resultClassifyMat = -1 * alphaClassifyMat * supportVectorClassifyMat;//����-(alphaMat * supportVectorMat),����ŵ�resultMat��  

	//-----------------------------����train��������׼ȷ��-------------------------
	//0.��������
	ifstream IrobotName2((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
	ifstream ObstacleName2((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
	ifstream BackgroundName2((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
	ifstream HardBackgroundName2((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 
	float TPtrain = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
		TNtrain = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FPtrain = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FNtrain = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
	float PrecisionTrain = -1,//Precision = TP/(TP+FP);
		RecallTrain = -1,//Recall = TP/(TP+FN);
		F1ScoreTrain = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)
	float TPtrain2 = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
		TNtrain2 = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FPtrain2 = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FNtrain2 = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
	float PrecisionTrain2 = -1,//Precision = TP/(TP+FP);
		RecallTrain2 = -1,//Recall = TP/(TP+FN);
		F1ScoreTrain2 = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)

	//������train������׼ȷ��
	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName2,ImgName); setIndex++)  
	{
		if (irobTypeArray[setIndex]==0)
		{
			ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				TPtrain = TPtrain + 1;
			} 
			else
			{
				FNtrain = FNtrain + 1;

			}
			if (classifyResult == 2)//������
			{
				FPtrain2 = FPtrain2 + 1;
			} 
			else
			{
				TNtrain2 = TNtrain2 + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName2,ImgName); setIndex++)  
	{
		if (obstTypeArray[setIndex]==0)
		{
			ImgName = ObstacleSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPtrain = FPtrain + 1;
			} 
			else
			{
				TNtrain = TNtrain + 1;
			}
			if (classifyResult == 2)//������
			{
				TPtrain2 = TPtrain2 + 1;
			} 
			else
			{
				FNtrain2 = FNtrain2 + 1;
			}
		}
	}
	//������train������׼ȷ��
	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName2,ImgName); setIndex++)  
	{
		if (backTypeArray[setIndex]==0)
		{
			ImgName = BackgroundSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPtrain = FPtrain + 1;
			} 
			else
			{
				TNtrain = TNtrain + 1;
			}
			if (classifyResult == 2)//������
			{
				FPtrain2 = FPtrain2 + 1;
			} 
			else
			{
				TNtrain2 = TNtrain2 + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName2,ImgName); setIndex++)  
	{
		if (hardTypeArray[setIndex]==0)
		{
			ImgName = HardBackgroundSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPtrain = FPtrain + 1;
			} 
			else
			{
				TNtrain = TNtrain + 1;
			}
			if (classifyResult == 2)//������
			{
				FPtrain2 = FPtrain2 + 1;
			} 
			else
			{
				TNtrain2 = TNtrain2 + 1;
			}
		}
	}
	//---------------------------------------end----------------------------------



	//-----------------------------����vaild��������׼ȷ��-------------------------
	//0.��������
	ifstream IrobotName3((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
	ifstream ObstacleName3((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
	ifstream BackgroundName3((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
	ifstream HardBackgroundName3((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 
	float TPvaild = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
		TNvaild = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FPvaild = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FNvaild = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
	float PrecisionVaild = -1,//Precision = TP/(TP+FP);
		RecallVaild = -1,//Recall = TP/(TP+FN);
		F1ScoreVaild = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)
	float TPvaild2 = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
		TNvaild2 = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FPvaild2 = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FNvaild2 = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
	float PrecisionVaild2 = -1,//Precision = TP/(TP+FP);
		RecallVaild2 = -1,//Recall = TP/(TP+FN);
		F1ScoreVaild2 = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)

	//������vaid������׼ȷ��
	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName3,ImgName); setIndex++)  
	{
		if (irobTypeArray[setIndex]==1)
		{
			ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				TPvaild = TPvaild + 1;
			} 
			else
			{
				FNvaild = FNvaild + 1;

			}
			if (classifyResult == 2)//������
			{
				FPvaild2 = FPvaild2 + 1;
			} 
			else
			{
				TNvaild2 = TNvaild2 + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName3,ImgName); setIndex++)  
	{
		if (obstTypeArray[setIndex]==1)
		{
			ImgName = ObstacleSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPvaild = FPvaild + 1;
			} 
			else
			{
				TNvaild = TNvaild + 1;
			}
			if (classifyResult == 2)//������
			{
				TPvaild2 = TPvaild2 + 1;
			} 
			else
			{
				FNvaild2 = FNvaild2 + 1;
			}
		}
	}
	//������vaild������׼ȷ��
	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName3,ImgName); setIndex++)  
	{
		if (backTypeArray[setIndex]==1)
		{
			ImgName = BackgroundSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPvaild = FPvaild + 1;
			} 
			else
			{
				TNvaild = TNvaild + 1;
			}
			if (classifyResult == 2)//������
			{
				FPvaild2 = FPvaild2 + 1;
			} 
			else
			{
				TNvaild2 = TNvaild2 + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName3,ImgName); setIndex++)  
	{
		if (hardTypeArray[setIndex]==1)
		{
			ImgName = HardBackgroundSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPvaild = FPvaild + 1;
			} 
			else
			{
				TNvaild = TNvaild + 1;
			}
			if (classifyResult == 2)//������
			{
				FPvaild2 = FPvaild2 + 1;
			} 
			else
			{
				TNvaild2 = TNvaild2 + 1;
			}
		}
	}
	//---------------------------------------end----------------------------------


	//-----------------------------����test��������׼ȷ��-------------------------
	//0.��������
	ifstream IrobotName4((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
	ifstream ObstacleName4((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
	ifstream BackgroundName4((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
	ifstream HardBackgroundName4((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 
	float TPtest = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
		TNtest = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FPtest = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FNtest = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
	float PrecisionTest = -1,//Precision = TP/(TP+FP);
		RecallTest = -1,//Recall = TP/(TP+FN);
		F1ScoreTest = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)
	float TPtest2 = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
		TNtest2 = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FPtest2 = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
		FNtest2 = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
	float PrecisionTest2 = -1,//Precision = TP/(TP+FP);
		RecallTest2 = -1,//Recall = TP/(TP+FN);
		F1ScoreTest2 = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)

	//������vaid������׼ȷ��
	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName4,ImgName); setIndex++)  
	{
		if (irobTypeArray[setIndex]==2)
		{
			ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				TPtest = TPtest + 1;
			} 
			else
			{
				FNtest = FNtest + 1;

			}
			if (classifyResult == 2)//������
			{
				FPtest2 = FPtest2 + 1;
			} 
			else
			{
				TNtest2 = TNtest2 + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName4,ImgName); setIndex++)  
	{
		if (obstTypeArray[setIndex]==2)
		{
			ImgName = ObstacleSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPtest = FPtest + 1;
			} 
			else
			{
				TNtest = TNtest + 1;
			}
			if (classifyResult == 2)//������
			{
				TPtest2 = TPtest2 + 1;
			} 
			else
			{
				FNtest2 = FNtest2 + 1;
			}
		}
	}
	//������test������׼ȷ��
	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName4,ImgName); setIndex++)  
	{
		if (backTypeArray[setIndex]==2)
		{
			ImgName = BackgroundSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPtest = FPtest + 1;
			} 
			else
			{
				TNtest = TNtest + 1;
			}
			if (classifyResult == 2)//������
			{
				FPtest2 = FPtest2 + 1;
			} 
			else
			{
				TNtest2 = TNtest2 + 1;
			}
		}
	}
	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName4,ImgName); setIndex++)  
	{
		if (hardTypeArray[setIndex]==2)
		{
			ImgName = HardBackgroundSetFile + ImgName;//���ϻ�����������·����  
			cout<<"����"<<ImgName<<endl;  
			Mat src = imread(ImgName);//��ȡͼƬ  
			resize(src,src,WinSizeClassify);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
			if(SHOWSET)//�Ƿ���ʾѵ������
			{
				imshow("��������������",src);
				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
			}

			vector<float> descriptors;//HOG����������
			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassfy, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
			classifyHOG.compute(src,descriptors);//����HOG������
			for(int i=0; i<descriptorDimClassfy; i++)  
				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
			if (classifyResult == 1)//������
			{
				FPtest = FPtest + 1;
			} 
			else
			{
				TNtest = TNtest + 1;
			}
			if (classifyResult == 2)//������
			{
				FPtest2 = FPtest2 + 1;
			} 
			else
			{
				TNtest2 = TNtest2 + 1;
			}
		}
	}
	//---------------------------------------end----------------------------------

	//---------------------------------ָ�����-----------------------------------
	cout<<"===============irobot==============="<<endl;
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
	cout<<endl;

	cout<<"===============obstacle==============="<<endl;
	PrecisionTrain2 = TPtrain2/(TPtrain2+FPtrain2);//Precision = TP/(TP+FP);
	RecallTrain2 = TPtrain2/(TPtrain2+FNtrain2);//Recall = TP/(TP+FN);
	F1ScoreTrain2 = 2*PrecisionTrain2*RecallTrain2/(PrecisionTrain2+RecallTrain2);//F1Score = 2*Precision*Recall/(Precision+Recall);
	cout<<"PrecisionTrain2:"<<PrecisionTrain2<<endl;
	cout<<"RecallTrain2:"<<RecallTrain2<<endl;
	cout<<"F1ScoreTrain2:"<<F1ScoreTrain2<<endl;
	cout<<endl;

	PrecisionVaild2 = TPvaild2/(TPvaild2+FPvaild2);//Precision = TP/(TP+FP);
	RecallVaild2 = TPvaild2/(TPvaild2+FNvaild2);//Recall = TP/(TP+FN);
	F1ScoreVaild2 = 2*PrecisionVaild2*RecallVaild2/(PrecisionVaild2+RecallVaild2);//F1Score = 2*Precision*Recall/(Precision+Recall);
	cout<<"PrecisionVaild2:"<<PrecisionVaild2<<endl;
	cout<<"RecallVaild2:"<<RecallVaild2<<endl;
	cout<<"F1ScoreVaild2:"<<F1ScoreVaild2<<endl;
	cout<<endl;

	PrecisionTest2 = TPtest2/(TPtest2+FPtest2);//Precision = TP/(TP+FP);
	RecallTest2 = TPtest2/(TPtest2+FNtest2);//Recall = TP/(TP+FN);
	F1ScoreTest2 = 2*PrecisionTest2*RecallTest2/(PrecisionTest2+RecallTest2);//F1Score = 2*Precision*Recall/(Precision+Recall);
	cout<<"PrecisionTest2:"<<PrecisionTest2<<endl;
	cout<<"RecallTest2:"<<RecallTest2<<endl;
	cout<<"F1ScoreTest2:"<<F1ScoreTest2<<endl;
	//---------------------------------------end----------------------------------
	getchar();
	return 0;

}  
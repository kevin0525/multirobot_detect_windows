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
//
////-----------------------������----------------------------
////---------------------------------------------------------
//
//int main()  
//{
//	//��������
//	HOGDescriptor detectHOG(WinSizeDetect,BlockSizeDetect,BlockStrideDetect,CellSizeDetect,NbinsDetect);//HOG�����ӣ���ⴰ�ڣ�block�ߴ磬block������cell�ߴ磬ֱ��ͼbin���� 
//	int descriptorDimDetect;//HOG�����ӵ�ά����[(��ⴰ�ڳ�-block��)/block����+1]*[(��ⴰ�ڸ�-block��)/block����+1]*bin����*(block��/cell��)*(block��/cell��)
//	MySVM detectSvm;//���SVM
//
//	//�ļ���������
//	string ImgName;//ͼƬ��
//	ifstream IrobotName((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
//	ifstream ObstacleName((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
//	ifstream BackgroundName((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
//	ifstream HardBackgroundName((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 
//
//	int irobTypeArray[IrobotSetNo];//��ʼ����������˳������
//	int obstTypeArray[ObstacleSetNo];
//	int backTypeArray[BackgroundSetNo];
//	int hardTypeArray[HardBackgroundSetNo];
//	random(irobTypeArray, IrobotSetNo);//������������˳������
//	random(obstTypeArray, ObstacleSetNo);
//	random(backTypeArray, BackgroundSetNo);
//	random(hardTypeArray, HardBackgroundSetNo);
//	typeHandle(irobTypeArray,IrobotSetNo,IrobotTrainNo,IrobotVaildNo);//�������͸�ֵ��train,vaild,test��
//	typeHandle(obstTypeArray,ObstacleSetNo,ObstacleTrainNo,ObstacleVaildNo);
//	typeHandle(backTypeArray,BackgroundSetNo,BackgroundTrainNo,BackgroundVaildNo);
//	typeHandle(hardTypeArray,HardBackgroundSetNo,HardBackgroundTrainNo,HardBackgroundVaildNo);
//
//
//	//----------------ѵ��������orֱ�Ӷ�ȡ������---------------------
//	//---------------------------------------------------------------
//	if(TRAIN) //ѵ����������������XML�ļ�
//	{
//		//ѵ����������
//		Mat sampleFeatureDetectMat;//ѵ��SVM������������������=��������������=��������ά��
//		Mat detectLabelMat;//���SVM�ĵ��������������=��������������=1��1��ʾ�л����˻��ϰ���-1��ʾ�޻����˺��ϰ�
//
//		//1.�������������ͼƬ
//		for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName,ImgName); setIndex++)  
//		{
//			if (irobTypeArray[setIndex]==0)
//			{
//				ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
//				cout<<"����"<<ImgName<<endl;  
//				Mat src = imread(ImgName);//��ȡͼƬ
//				Mat srcDetect;
//				resize(src,srcDetect,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//				if(SHOWSET)//�Ƿ���ʾѵ������
//				{
//					imshow("��������������",src);
//					if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//				}
//
//				vector<float> descriptorsDetect;//HOG����������
//				detectHOG.compute(srcDetect,descriptorsDetect);//����HOG������
//
//				if(num == 0)//�����һ������ʱ��ʼ���������������������  
//				{
//					descriptorDimDetect = descriptorsDetect.size();//HOG�����ӵ�ά��
//					sampleFeatureDetectMat = Mat::zeros(AllTrainNo, descriptorDimDetect, CV_32FC1);
//					detectLabelMat = Mat::zeros(AllTrainNo, 1, CV_32FC1);  
//				} 
//
//				//������õ�HOG�����Ӹ��Ƶ��������������������
//				for(int i=0; i<descriptorDimDetect; i++)
//					sampleFeatureDetectMat.at<float>(num,i) = descriptorsDetect[i];//��num�����������������еĵ�i��Ԫ��  
//				detectLabelMat.at<float>(num,0) = 1;//1��ʾ�л�����
//				num++;
//			}
//		}  
//
//		//2.�����ϰ�����ͼƬ
//		for(int setIndex=0,num=IrobotTrainNo; setIndex<ObstacleSetNo && getline(ObstacleName,ImgName); setIndex++)  
//		{
//			if (obstTypeArray[setIndex]==0)
//			{
//				ImgName = ObstacleSetFile + ImgName;//�����ϰ�������·����
//				cout<<"����"<<ImgName<<endl;  
//				Mat src = imread(ImgName);//��ȡͼƬ
//				Mat srcDetect;
//				resize(src,srcDetect,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//				if(SHOWSET)//�Ƿ���ʾѵ������
//				{
//					imshow("��������������",src);
//					if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//				}
//
//				vector<float> descriptorsDetect;//HOG����������
//				detectHOG.compute(srcDetect,descriptorsDetect);//����HOG������
//
//				//������õ�HOG�����Ӹ��Ƶ��������������������
//				for(int i=0; i<descriptorDimDetect; i++)
//					sampleFeatureDetectMat.at<float>(num,i) = descriptorsDetect[i];//��num�����������������еĵ�i��Ԫ��  
//				detectLabelMat.at<float>(num,0) = -1;//1��ʾ�л�����
//				num++;
//			}
//		}
//
//		//3.����������ͼƬ
//		for(int setIndex=0,num=IrobotTrainNo+ObstacleTrainNo; setIndex<BackgroundSetNo && getline(BackgroundName,ImgName); setIndex++)  
//		{
//			if (backTypeArray[setIndex]==0)
//			{
//				ImgName = BackgroundSetFile + ImgName;//���ϱ���������·����  
//				cout<<"����"<<ImgName<<endl;  
//				Mat src = imread(ImgName);//��ȡͼƬ
//				Mat srcDetect;
//				resize(src,srcDetect,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//				if(SHOWSET)//�Ƿ���ʾѵ������
//				{
//					imshow("��������������",src);
//					if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//				}
//
//				vector<float> descriptorsDetect;//HOG����������
//				detectHOG.compute(srcDetect,descriptorsDetect);//����HOG������
//
//				//������õ�HOG�����Ӹ��Ƶ��������������������
//				for(int i=0; i<descriptorDimDetect; i++)
//					sampleFeatureDetectMat.at<float>(num,i) = descriptorsDetect[i];//��num�����������������еĵ�i��Ԫ��  
//				detectLabelMat.at<float>(num,0) = -1;//1��ʾ�л�����
//				num++;
//			}
//		}
//
//		//4.����Hard��������ͼƬ
//		for(int setIndex=0,num=IrobotTrainNo+ObstacleTrainNo+BackgroundTrainNo; setIndex<HardBackgroundSetNo && getline(HardBackgroundName,ImgName); setIndex++)  
//		{
//			if (hardTypeArray[setIndex]==0)
//			{
//				ImgName = HardBackgroundSetFile + ImgName;//����Hard����������·����
//				cout<<"����"<<ImgName<<endl;
//				Mat src = imread(ImgName);//��ȡͼƬ
//				Mat srcDetect;
//				resize(src,srcDetect,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//				if(SHOWSET)//�Ƿ���ʾѵ������
//				{
//					imshow("��������������",src);
//					if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//				}
//
//				vector<float> descriptorsDetect;//HOG����������
//				detectHOG.compute(srcDetect,descriptorsDetect);//����HOG������
//
//				//������õ�HOG�����Ӹ��Ƶ��������������������
//				for(int i=0; i<descriptorDimDetect; i++)
//					sampleFeatureDetectMat.at<float>(num,i) = descriptorsDetect[i];//��num�����������������еĵ�i��Ԫ��  
//				detectLabelMat.at<float>(num,0) = -1;//1��ʾ�л���
//				num++;
//			}
//		}  
//
//		//5.ѵ�����SVM������  
//		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
//		CvTermCriteria detectCriteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);  
//		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
//		CvSVMParams detectParam(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, detectCriteria);  
//		cout<<"��ʼѵ�����SVM������"<<endl;  
//		detectSvm.train(sampleFeatureDetectMat, detectLabelMat, Mat(), Mat(), detectParam);//ѵ��������
//		cout<<"ѵ�����"<<endl;  
//		detectSvm.save(DetectSvmName);//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�  
//	}  
//	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����  
//	{  
//		detectSvm.load(DetectSvmName);
//	}  
//
//	//----------------���л����˺��ϰ���ļ�������-----------------
//	//---------------------------------------------------------------
//	//��������
//	descriptorDimDetect = detectSvm.get_var_count();//����������ά������HOG�����ӵ�ά������ǰ��ѵ��ʱ�Ĵ�Сһ������Ӵ˾���Ϊ���ڲ�ѵ��ʱҲ���õ�ά����
//	int supportVectorNumDetect = detectSvm.get_support_vector_count();//֧�������ĸ���
//	cout<<"Detect֧������������"<<supportVectorNumDetect<<endl;  
//	Mat alphaDetectMat = Mat::zeros(1, supportVectorNumDetect, CV_32FC1);//alpha���������ȵ���֧����������
//	Mat supportVectorDetectMat = Mat::zeros(supportVectorNumDetect, descriptorDimDetect, CV_32FC1);//֧���������� 
//	Mat resultDetectMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//alpha��������֧����������Ľ�� 
//
//	//����w����
//	for(int i=0; i<supportVectorNumDetect; i++)//��֧�����������ݸ��Ƶ�supportVectorMat������  
//	{
//		const float * pSVData = detectSvm.get_support_vector(i);//���ص�i��֧������������ָ��  
//		for(int j=0; j<descriptorDimDetect; j++)  
//			supportVectorDetectMat.at<float>(i,j) = pSVData[j];  
//	}
//
//	double * pAlphaDataDetect = detectSvm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
//	for(int i=0; i<supportVectorNumDetect; i++)//��alpha���������ݸ��Ƶ�alphaMat��  
//		alphaDetectMat.at<float>(0,i) = pAlphaDataDetect[i];  
//	resultDetectMat = -1 * alphaDetectMat * supportVectorDetectMat;//����-(alphaMat * supportVectorMat),����ŵ�resultMat�� 
//
//	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����  
//	vector<float> myDetector;//����Hog������SVM����ӣ�w+b��
//	for(int i=0; i<descriptorDimDetect; i++)//��resultMat�е����ݸ��Ƶ�����myDetector��  
//		myDetector.push_back(resultDetectMat.at<float>(0,i));  
//	myDetector.push_back(detectSvm.get_rho());//������ƫ����rho���õ������  
//	cout<<"����Hog������SVM�����ά��(w+b)��"<<myDetector.size()<<endl;
//
//	//����SVMDetector�����
//	detectHOG.setSVMDetector(myDetector);  
//
//	//-----------------------------����train��������׼ȷ��-------------------------
//	//0.��������
//	ifstream IrobotName2((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
//	ifstream ObstacleName2((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
//	ifstream BackgroundName2((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
//	ifstream HardBackgroundName2((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 
//	float TPtrain = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
//		TNtrain = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
//		FPtrain = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
//		FNtrain = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
//	float PrecisionTrain = -1,//Precision = TP/(TP+FP);
//		RecallTrain = -1,//Recall = TP/(TP+FN);
//		F1ScoreTrain = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)
//
//	//������train������׼ȷ��
//	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName2,ImgName); setIndex++)  
//	{
//		if (irobTypeArray[setIndex]==0)
//		{
//			ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				TPtrain = TPtrain + 1;
//			} 
//			else
//			{
//				FNtrain = FNtrain + 1;
//			}
//		}
//	}
//	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName2,ImgName); setIndex++)  
//	{
//		if (obstTypeArray[setIndex]==0)
//		{
//			ImgName = ObstacleSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPtrain = FPtrain + 1;
//			} 
//			else
//			{
//				TNtrain = TNtrain + 1;
//			}
//		}
//	}
//	//������train������׼ȷ��
//	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName2,ImgName); setIndex++)  
//	{
//		if (backTypeArray[setIndex]==0)
//		{
//			ImgName = BackgroundSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPtrain = FPtrain + 1;
//			} 
//			else
//			{
//				TNtrain = TNtrain + 1;
//			}
//		}
//	}
//	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName2,ImgName); setIndex++)  
//	{
//		if (hardTypeArray[setIndex]==0)
//		{
//			ImgName = HardBackgroundSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPtrain = FPtrain + 1;
//			} 
//			else
//			{
//				TNtrain = TNtrain + 1;
//			}
//		}
//	}
//	//---------------------------------------end----------------------------------
//
//
//
//	//-----------------------------����vaild��������׼ȷ��-------------------------
//	//0.��������
//	ifstream IrobotName3((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
//	ifstream ObstacleName3((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
//	ifstream BackgroundName3((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
//	ifstream HardBackgroundName3((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 
//	float TPvaild = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
//		TNvaild = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
//		FPvaild = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
//		FNvaild = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
//	float PrecisionVaild = -1,//Precision = TP/(TP+FP);
//		RecallVaild = -1,//Recall = TP/(TP+FN);
//		F1ScoreVaild = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)
//
//	//������vaid������׼ȷ��
//	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName3,ImgName); setIndex++)  
//	{
//		if (irobTypeArray[setIndex]==1)
//		{
//			ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				TPvaild = TPvaild + 1;
//			} 
//			else
//			{
//				FNvaild = FNvaild + 1;
//			}
//		}
//	}
//	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName3,ImgName); setIndex++)  
//	{
//		if (obstTypeArray[setIndex]==1)
//		{
//			ImgName = ObstacleSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPvaild = FPvaild + 1;
//			} 
//			else
//			{
//				TNvaild = TNvaild + 1;
//			}
//		}
//	}
//	//������vaild������׼ȷ��
//	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName3,ImgName); setIndex++)  
//	{
//		if (backTypeArray[setIndex]==1)
//		{
//			ImgName = BackgroundSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPvaild = FPvaild + 1;
//			} 
//			else
//			{
//				TNvaild = TNvaild + 1;
//			}
//		}
//	}
//	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName3,ImgName); setIndex++)  
//	{
//		if (hardTypeArray[setIndex]==1)
//		{
//			ImgName = HardBackgroundSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPvaild = FPvaild + 1;
//			} 
//			else
//			{
//				TNvaild = TNvaild + 1;
//			}
//		}
//	}
//	//---------------------------------------end----------------------------------
//
//	//-----------------------------����test��������׼ȷ��-------------------------
//	//0.��������
//	ifstream IrobotName4((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
//	ifstream ObstacleName4((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
//	ifstream BackgroundName4((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
//	ifstream HardBackgroundName4((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 
//	float TPtest = 0,//TP:ʵ��Ϊ�棬Ԥ��Ϊ��
//		TNtest = 0,//TN:ʵ��Ϊ�٣�Ԥ��Ϊ��
//		FPtest = 0,//FP:ʵ��Ϊ�٣�Ԥ��Ϊ��
//		FNtest = 0;//FN:ʵ��Ϊ�棬Ԥ��Ϊ��
//	float PrecisionTest = -1,//Precision = TP/(TP+FP);
//		RecallTest = -1,//Recall = TP/(TP+FN);
//		F1ScoreTest = -1;//F1Score = 2*Precision*Recall/(Precision+Recall)
//
//	//������vaid������׼ȷ��
//	for(int setIndex=0,num=0; setIndex<IrobotSetNo && getline(IrobotName4,ImgName); setIndex++)  
//	{
//		if (irobTypeArray[setIndex]==2)
//		{
//			ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				TPtest = TPtest + 1;
//			} 
//			else
//			{
//				FNtest = FNtest + 1;
//			}
//		}
//	}
//	for(int setIndex=0,num=0; setIndex<ObstacleSetNo && getline(ObstacleName4,ImgName); setIndex++)  
//	{
//		if (obstTypeArray[setIndex]==2)
//		{
//			ImgName = ObstacleSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPtest = FPtest + 1;
//			} 
//			else
//			{
//				TNtest = TNtest + 1;
//			}
//		}
//	}
//	//������test������׼ȷ��
//	for(int setIndex=0,num=0; setIndex<BackgroundSetNo && getline(ObstacleName4,ImgName); setIndex++)  
//	{
//		if (backTypeArray[setIndex]==2)
//		{
//			ImgName = BackgroundSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPtest = FPtest + 1;
//			} 
//			else
//			{
//				TNtest = TNtest + 1;
//			}
//		}
//	}
//	for(int setIndex=0,num=0; setIndex<HardBackgroundSetNo && getline(HardBackgroundName4,ImgName); setIndex++)  
//	{
//		if (hardTypeArray[setIndex]==2)
//		{
//			ImgName = HardBackgroundSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSizeDetect);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			detectHOG.compute(src,descriptors);//����HOG������
//			for(int i=0; i<descriptorDimDetect; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//			float detectResult = detectSvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//			if (detectResult == 1)//������
//			{
//				FPtest = FPtest + 1;
//			} 
//			else
//			{
//				TNtest = TNtest + 1;
//			}
//		}
//	}
//	//---------------------------------------end----------------------------------
//
//	//---------------------------------ָ�����-----------------------------------
//	PrecisionTrain = TPtrain/(TPtrain+FPtrain);//Precision = TP/(TP+FP);
//	RecallTrain = TPtrain/(TPtrain+FNtrain);//Recall = TP/(TP+FN);
//	F1ScoreTrain = 2*PrecisionTrain*RecallTrain/(PrecisionTrain+RecallTrain);//F1Score = 2*Precision*Recall/(Precision+Recall);
//	cout<<"PrecisionTrain:"<<PrecisionTrain<<endl;
//	cout<<"RecallTrain:"<<RecallTrain<<endl;
//	cout<<"F1ScoreTrain:"<<F1ScoreTrain<<endl;
//	cout<<endl;
//
//	PrecisionVaild = TPvaild/(TPvaild+FPvaild);//Precision = TP/(TP+FP);
//	RecallVaild = TPvaild/(TPvaild+FNvaild);//Recall = TP/(TP+FN);
//	F1ScoreVaild = 2*PrecisionVaild*RecallVaild/(PrecisionVaild+RecallVaild);//F1Score = 2*Precision*Recall/(Precision+Recall);
//	cout<<"PrecisionVaild:"<<PrecisionVaild<<endl;
//	cout<<"RecallVaild:"<<RecallVaild<<endl;
//	cout<<"F1ScoreVaild:"<<F1ScoreVaild<<endl;
//	cout<<endl;
//
//	PrecisionTest = TPtest/(TPtest+FPtest);//Precision = TP/(TP+FP);
//	RecallTest = TPtest/(TPtest+FNtest);//Recall = TP/(TP+FN);
//	F1ScoreTest = 2*PrecisionTest*RecallTest/(PrecisionTest+RecallTest);//F1Score = 2*Precision*Recall/(Precision+Recall);
//	cout<<"PrecisionTest:"<<PrecisionTest<<endl;
//	cout<<"RecallTest:"<<RecallTest<<endl;
//	cout<<"F1ScoreTest:"<<F1ScoreTest<<endl;
//	//---------------------------------------end----------------------------------
//	getchar();
//	return 0;
//}  
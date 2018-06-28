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
////-----------------------�궨��----------------------------
////---------------------------------------------------------
//#define IrobotSetNo 218			//��������������  
//#define ObstacleSetNo 205		//�ϰ���������
//#define BackgroundSetNo 736		//������������
//#define HardBackgroundSetNo 155	//Hard������������
//
//#define SHOWSET false			//�Ƿ���ʾѵ������
//#define TRAIN false				//�Ƿ����ѵ��,true��ʾѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
//#define SAVESET false			//�Ƿ񱣴�������
//
////HOG�����Ӳ���
//#define WinSize Size(40,20)		//��ⴰ�ڳߴ�
//#define BlockSize Size(8,8)		//block�ߴ�
//#define BlockStride Size(4,4)	//block����
//#define CellSize Size(4,4)		//cell�ߴ�
//#define Nbins 9					//ֱ��ͼbin����
//
//////HOG�����Ӳ���
////#define WinSize Size(32,16)		//��ⴰ�ڳߴ�
////#define BlockSize Size(8,8)		//block�ߴ�
////#define BlockStride Size(4,4)	//block����
////#define CellSize Size(4,4)		//cell�ߴ�
////#define Nbins 9					//ֱ��ͼbin����
//
//////HOG�����Ӳ���
////#define WinSize Size(20,10)		//��ⴰ�ڳߴ�
////#define BlockSize Size(4,4)		//block�ߴ�
////#define BlockStride Size(2,2)	//block����
////#define CellSize Size(2,2)		//cell�ߴ�
////#define Nbins 9					//ֱ��ͼbin����
//
//////HOG�����Ӳ���
////#define WinSize Size(64,32)		//��ⴰ�ڳߴ�
////#define BlockSize Size(8,8)		//block�ߴ�
////#define BlockStride Size(4,4)		//block����
////#define CellSize Size(4,4)		//cell�ߴ�
////#define Nbins 9					//ֱ��ͼbin����
//
////detectMultiScale���ֲ���
//#define HitThreshold 0			//���������볬ƽ����С����
//#define WinStride Size(4,4)		//�ƶ�����(������block������������)
//#define DetScale 1.05			//Դͼ��ͼ��ÿ����С����
//
//#define TestImage "../Data/TestImage/13.jpg"				//���ڼ��Ĳ���ͼ��
//#define ResultImage "../Data/Result/13.jpg"					//����ͼ��ļ����
//#define ResultImageFile_1 "..\\Data\\Result\\13-1\\"		//����ͼ��ķ����ͼ�ļ���1
//#define ResultImageFile_2 "..\\Data\\Result\\13-2\\"		//����ͼ��ķ����ͼ�ļ���2
//#define ResultImageFile_3 "..\\Data\\Result\\13-3\\"		//����ͼ��ķ����ͼ�ļ���3
//#define TestVideo "../Data/TestVideo/1��-���.avi"			//���ڼ��Ĳ�����Ƶ
//#define ResultVideo "../Data/Result/1��-���.avi"			//������Ƶ�ļ����
//#define ResultVideoFile_1 "..\\Data\\Result\\1��-���-1\\"	//������Ƶ�ķ����ͼ�ļ���1
//#define ResultVideoFile_2 "..\\Data\\Result\\1��-���-2\\"	//������Ƶ�ķ����ͼ�ļ���2
//#define ResultVideoFile_3 "..\\Data\\Result\\1��-���-3\\"	//������Ƶ�ķ����ͼ�ļ���3
//
//#define IrobotSetFile "../Data/IrobotSet/"					//����������ͼƬ�ļ���
//#define ObstacleSetFile "../Data/ObstacleSet/"				//�ϰ�����ͼƬ�ļ���
//#define BackgroundSetFile "../Data/BackgroundSet/"			//��������ͼƬ�ļ���
//#define HardBackgroundSetFile "../Data/HardBackgroundSet/"	//Hard����ͼƬ�ļ���
//#define SetName "0SetName.txt"								//����ͼƬ���ļ����б�txt
//
//#define DetectSvmName "../Data/Result/SVM_HOG_Detect.xml"	//�������ȡ�ļ��ģ���ļ�����
//#define ClassifySvmName "../Data/Result/SVM_HOG_Classify.xml"//�������ȡ�ķ���ģ���ļ�����
//
////-----------------------������----------------------------
////---------------------------------------------------------
//
//int main()  
//{
//	//��������
//    HOGDescriptor hog(WinSize,BlockSize,BlockStride,CellSize,Nbins);//HOG�����ӣ���ⴰ�ڣ�block�ߴ磬block������cell�ߴ磬ֱ��ͼbin���� 
//    int descriptorDim;//HOG�����ӵ�ά����[(��ⴰ�ڳ�-block��)/block����+1]*[(��ⴰ�ڸ�-block��)/block����+1]*cell��*cell��*bin����
//    MySVM detectSvm;//���SVM
//	MySVM classifySvm;//����SVM
//
//    //----------------ѵ��������orֱ�Ӷ�ȡ������---------------------
//	//---------------------------------------------------------------
//    if(TRAIN) //ѵ����������������XML�ļ�
//    {
//		//ѵ����������
//        string ImgName;//ͼƬ��
//        ifstream IrobotName((string)IrobotSetFile+SetName);//����������ͼƬ���ļ����б�
//		ifstream ObstacleName((string)ObstacleSetFile+SetName);//�ϰ�����ͼƬ���ļ����б�
//        ifstream BackgroundName((string)BackgroundSetFile+SetName);//��������ͼƬ���ļ����б�
//		ifstream HardBackgroundName((string)HardBackgroundSetFile+SetName);//Hard��������ͼƬ���ļ����б� 
//        Mat sampleFeatureMat;//ѵ��SVM������������������=��������������=��������ά��
//        Mat detectLabelMat;//���SVM�ĵ��������������=��������������=1��1��ʾ�л����˻��ϰ���-1��ʾ�޻����˺��ϰ�
//		Mat classifyLabelMat;//����SVM�ĵ��������������=��������������=1��1��ʾ�л����ˣ�2��ʾ���ϰ���3��ʾ�б���
//  
//        //1.�������������ͼƬ
//        for(int num=0; num<IrobotSetNo && getline(IrobotName,ImgName); num++)  
//        {  
//            ImgName = IrobotSetFile + ImgName;//���ϻ�����������·����  
//			cout<<"����"<<ImgName<<endl;  
//            Mat src = imread(ImgName);//��ȡͼƬ  
//			resize(src,src,WinSize);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//			
//            vector<float> descriptors;//HOG����������  
//            hog.compute(src,descriptors);//����HOG������
//  
//            if(num == 0)//�����һ������ʱ��ʼ���������������������  
//            {
//                descriptorDim = descriptors.size();//HOG�����ӵ�ά��  
//                sampleFeatureMat = Mat::zeros(IrobotSetNo+ObstacleSetNo+BackgroundSetNo+HardBackgroundSetNo, descriptorDim, CV_32FC1);  
//                detectLabelMat = Mat::zeros(IrobotSetNo+ObstacleSetNo+BackgroundSetNo+HardBackgroundSetNo, 1, CV_32FC1);  
//				classifyLabelMat = Mat::zeros(IrobotSetNo+ObstacleSetNo+BackgroundSetNo+HardBackgroundSetNo, 1, CV_32FC1);  
//            } 
//
//            //������õ�HOG�����Ӹ��Ƶ��������������������
//            for(int i=0; i<descriptorDim; i++)
//                sampleFeatureMat.at<float>(num,i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
//            detectLabelMat.at<float>(num,0) = 1;//1��ʾ�л����˻��ϰ� 
//			classifyLabelMat.at<float>(num,0) = 1;//1��ʾ�л�����
//        }  
//
//		//2.�����ϰ�����ͼƬ
//		for(int num=IrobotSetNo; num<IrobotSetNo+ObstacleSetNo && getline(ObstacleName,ImgName); num++)  
//		{ 
//			ImgName = ObstacleSetFile + ImgName;//�����ϰ�������·����
//			cout<<"����"<<ImgName<<endl;  
//			Mat src = imread(ImgName);//��ȡͼƬ 
//			resize(src,src,WinSize);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//			vector<float> descriptors;//HOG����������  
//			hog.compute(src,descriptors);//����HOG������
//
//			//������õ�HOG�����Ӹ��Ƶ��������������������
//			for(int i=0; i<descriptorDim; i++)  
//				sampleFeatureMat.at<float>(num,i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
//			detectLabelMat.at<float>(num,0) = 1;//1��ʾ�л����˻��ϰ�
//			classifyLabelMat.at<float>(num,0) = 2;//2��ʾ���ϰ�
//		}
//
//        //3.����������ͼƬ
//        for(int num=IrobotSetNo+ObstacleSetNo; num<IrobotSetNo+ObstacleSetNo+BackgroundSetNo && getline(BackgroundName,ImgName); num++)  
//        {  
//            ImgName = BackgroundSetFile + ImgName;//���ϱ���������·����  
//			cout<<"����"<<ImgName<<endl;  
//            Mat src = imread(ImgName);//��ȡͼƬ 
//			resize(src,src,WinSize);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("������������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//            vector<float> descriptors;//HOG����������  
//            hog.compute(src,descriptors);//����HOG������
//  
//            //������õ�HOG�����Ӹ��Ƶ��������������������
//            for(int i=0; i<descriptorDim; i++)  
//                sampleFeatureMat.at<float>(num,i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
//            detectLabelMat.at<float>(num,0) = -1;//-1��ʾ�޻����˺��ϰ�
//			classifyLabelMat.at<float>(num,0) = 3;//3��ʾ�б���
//        }
//
//        //4.����Hard��������ͼƬ
//        for(int num=IrobotSetNo+ObstacleSetNo+BackgroundSetNo; num<IrobotSetNo+ObstacleSetNo+BackgroundSetNo+HardBackgroundSetNo && getline(HardBackgroundName,ImgName); num++)  
//        {
//            ImgName = HardBackgroundSetFile + ImgName;//����Hard����������·����  
//			cout<<"����"<<ImgName<<endl;
//            Mat src = imread(ImgName);//��ȡͼƬ 
//			resize(src,src,WinSize);//��ѵ��������һ��Ϊ��ⴰ�ڵĴ�С
//			if(SHOWSET)//�Ƿ���ʾѵ������
//			{
//				imshow("��HardExample����������",src);
//				if (waitKey(1)>0){return 0;}//ͨ�������жϳ���
//			}
//
//            vector<float> descriptors;//HOG����������  
//            hog.compute(src,descriptors);//����HOG������
//  
//            //������õ�HOG�����Ӹ��Ƶ�������������������� 
//            for(int i=0; i<descriptorDim; i++)  
//                sampleFeatureMat.at<float>(num,i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
//            detectLabelMat.at<float>(num,0) = -1;//�����������Ϊ-1���޻�����
//			classifyLabelMat.at<float>(num,0) = 3;//3��ʾ�б���
//        }  
//
//        //5.ѵ�����SVM������  
//        //������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
//        CvTermCriteria detectCriteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);  
//        //SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
//        CvSVMParams detectParam(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, detectCriteria);  
//        cout<<"��ʼѵ�����SVM������"<<endl;  
//        detectSvm.train(sampleFeatureMat, detectLabelMat, Mat(), Mat(), detectParam);//ѵ��������
//        cout<<"ѵ�����"<<endl;  
//        detectSvm.save(DetectSvmName);//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�  
//
//		//6.ѵ������SVM������  
//		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
//		CvTermCriteria classifyCriteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);  
//		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
//		CvSVMParams classifyParam(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, classifyCriteria);  
//		cout<<"��ʼѵ������SVM������"<<endl;  
//		classifySvm.train(sampleFeatureMat, classifyLabelMat, Mat(), Mat(), classifyParam);//ѵ��������
//		cout<<"ѵ�����"<<endl;  
//		classifySvm.save(ClassifySvmName);//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�  
//    }  
//    else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����  
//    {  
//        detectSvm.load(DetectSvmName);
//		classifySvm.load(ClassifySvmName);
//    }  
//
//	//----------------���л����˺��ϰ���ļ�������-----------------
//	//---------------------------------------------------------------
//	//��������
//	HOGDescriptor detectHOG(WinSize,BlockSize,BlockStride,CellSize,Nbins);//����HOG�����
//	HOGDescriptor classifyHOG(WinSize,BlockSize,BlockStride,CellSize,Nbins);//���HOG�����ڼ���������HOG��������
//    descriptorDim = detectSvm.get_var_count();//����������ά������HOG�����ӵ�ά������ǰ��ѵ��ʱ�Ĵ�Сһ������Ӵ˾���Ϊ���ڲ�ѵ��ʱҲ���õ�ά����
//    int supportVectorNum = detectSvm.get_support_vector_count();//֧�������ĸ���  
//    cout<<"֧������������"<<supportVectorNum<<endl;  
//    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������  
//    Mat supportVectorMat = Mat::zeros(supportVectorNum, descriptorDim, CV_32FC1);//֧����������  
//    Mat resultMat = Mat::zeros(1, descriptorDim, CV_32FC1);//alpha��������֧����������Ľ��  
//
//    //����w����
//    for(int i=0; i<supportVectorNum; i++)//��֧�����������ݸ��Ƶ�supportVectorMat������  
//	{
//        const float * pSVData = detectSvm.get_support_vector(i);//���ص�i��֧������������ָ��  
//        for(int j=0; j<descriptorDim; j++)  
//            supportVectorMat.at<float>(i,j) = pSVData[j];  
//    }
//    double * pAlphaData = detectSvm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
//    for(int i=0; i<supportVectorNum; i++)//��alpha���������ݸ��Ƶ�alphaMat��  
//        alphaMat.at<float>(0,i) = pAlphaData[i];  
//    resultMat = -1 * alphaMat * supportVectorMat;//����-(alphaMat * supportVectorMat),����ŵ�resultMat�� 
//
//    //�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����  
//    vector<float> myDetector;//����Hog������SVM����ӣ�w+b��
//    for(int i=0; i<descriptorDim; i++)//��resultMat�е����ݸ��Ƶ�����myDetector��  
//        myDetector.push_back(resultMat.at<float>(0,i));  
//    myDetector.push_back(detectSvm.get_rho());//������ƫ����rho���õ������  
//    cout<<"����Hog������SVM�����ά��(w+b)��"<<myDetector.size()<<endl;
//
//	//����SVMDetector�����
//    detectHOG.setSVMDetector(myDetector);  
//
//
//
// //   //-------------------------����ͼƬ���м�������-----------------------------------
//	////��������
// //   Mat src = imread(TestImage); //��ȡ����ͼ��
//	////resize(src,src,Size(0,0),2,2);//��������ͼ��Ĵ�С
//	//Mat dst; //���ͼ��
//	//src.copyTo(dst);
//	//vector<Rect> found, found_filtered;//��������
//	//const char * sd1 = {"md "ResultImageFile_1};//������ż���ͼ���ļ���
//	//system(sd1);
//	//const char * sd2 = {"md "ResultImageFile_2};//������ż���ͼ���ļ���
//	//system(sd2);
//	//const char * sd3 = {"md "ResultImageFile_3};//������ż���ͼ���ļ���
//	//system(sd3);
//	//
//	////��ͼƬ���ж�߶Ȼ����˼�� 
// //   cout<<"���ж�߶ȼ��"<<endl;
// //   detectHOG.detectMultiScale(src, found, HitThreshold, WinStride, Size(0,0), DetScale, 2, false);
//	//	//������1Դͼ��2���������3���������ͳ�ƽ��ľ���4�ƶ�����(������block������������)5��Ե��չ6Դͼ��ͼ��ÿ����С����7�������8���෽ʽ
// //   for(int i=0; i < found.size(); i++)  //�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
// //   {  
// //       Rect r = found[i];  
// //       int j=0;  
// //       for(; j < found.size(); j++)  
// //           if(j != i && (r & found[j]) == r)  
// //               break;  
// //       if( j == found.size())  
// //           found_filtered.push_back(r);  
// //   }
//	//cout<<"�ҵ��ľ��ο������"<<found_filtered.size()<<endl;
//	//
//	////�Լ�⵽��ͼ����з���
//	//for(int i=0; i<found_filtered.size(); i++)  
//	//{
//	//	cout<<"width:"<<found_filtered[i].width<<"  height:"<<found_filtered[i].height<<endl;//�������ͼ��С
//	//	vector<float> descriptors;//HOG����������
//	//	Mat descriptorsMat = Mat::zeros(1, descriptorDim, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//	//	Mat temp;
//	//	resize(src(found_filtered[i]),temp,WinSize);//���������ͼ��ߴ�
//
//	//	classifyHOG.compute(temp,descriptors);//����HOG������
//	//	for(int i=0; i<descriptorDim; i++)  
//	//		descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//
//	//	float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//
//	//	if (classifyResult == 1)//������
//	//	{
//	//		rectangle(dst, found_filtered[i], Scalar(255,0,0), 3);//��ͼ�л�������
//	//		if (SAVESET)//�Ƿ񱣴�������
//	//		{
//	//			strstream ss;
//	//			string s;
//	//			ss<<ResultImageFile_1<<i<<".jpg";
//	//			ss>>s;
//	//			imwrite(s,src(found_filtered[i]));
//	//		}
//	//	} 
//	//	else if (classifyResult == 2)//�ϰ���
//	//	{
//	//		rectangle(dst, found_filtered[i], Scalar(0,255,0), 3);//��ͼ�л�������
//	//		if (SAVESET)//�Ƿ񱣴�������
//	//		{
//	//			strstream ss;
//	//			string s;
//	//			ss<<ResultImageFile_2<<i<<".jpg";
//	//			ss>>s;
//	//			imwrite(s,src(found_filtered[i]));
//	//		}
//	//	}
//	//	else if (classifyResult ==3)//����
//	//	{
//	//		rectangle(dst, found_filtered[i], Scalar(0,0,255), 3);//��ͼ�л�������
//	//		if (SAVESET)//�Ƿ񱣴�������
//	//		{
//	//			strstream ss;
//	//			string s;
//	//			ss<<ResultImageFile_3<<i<<".jpg";
//	//			ss>>s;
//	//			imwrite(s,src(found_filtered[i]));
//	//		}
//	//	}
//	//	else//����
//	//	{
//	//		rectangle(dst, found_filtered[i], Scalar(255,255,255), 3);//��ͼ�л�������
//	//	}
//
//	//}
//
//	////������ͼ����
// //   imwrite(ResultImage,dst);  
// //   namedWindow("dst");  
// //   imshow("dst",dst);  
// //   waitKey(0);//ע�⣺imshow֮������waitKey�������޷���ʾͼ��  
//
//	////---------------------------------end---------------------------------------
//
//
//
//	//-------------------------������Ƶ���л����˼��-----------------------------------
//	//��������
//	VideoCapture myVideo(TestVideo);//��ȡ��Ƶ  
//	Mat src,dst;					//ԭʼͼ�񣬴����ͼ��
//	
//	const char * sd1 = {"md "ResultVideoFile_1};//������ż���ͼ���ļ���
//	system(sd1);
//	const char * sd2 = {"md "ResultVideoFile_2};//������ż���ͼ���ļ���
//	system(sd2);
//	const char * sd3 = {"md "ResultVideoFile_3};//������ż���ͼ���ļ���
//	system(sd3);
//
//	//����Ƶ
//	if(!myVideo.isOpened()){cout<<"��Ƶ��ȡ����"<<endl;system("puase");return -1;}
//
//	//�������ɵ���Ƶ
//	double videoRate=myVideo.get(CV_CAP_PROP_FPS);//��ȡ֡��
//	int videoWidth=myVideo.get(CV_CAP_PROP_FRAME_WIDTH);//��ȡ��Ƶͼ����
//	int videoHight=myVideo.get(CV_CAP_PROP_FRAME_HEIGHT);//��ȡ��Ƶͼ��߶�
//	int videoDelay=1000/videoRate;//ÿ֮֡����ӳ�����Ƶ��֡�����Ӧ�������ܳ����ʱ�򲥷���Ƶ�����ʣ�
//	VideoWriter outputVideo(ResultVideo, CV_FOURCC('M', 'J', 'P', 'G'), videoRate, Size(videoWidth, videoHight));//������Ƶ��
//
//	//��ʼ��Ƶ����
//	bool stop = false;
//	for (int fnum = 1;!stop;fnum++)
//	{
//		//��������
//		if (!myVideo.read(src)){cout<<"��Ƶ����"<<endl;waitKey(0); break;}//��ȡ��Ƶ֡
//		//resize(videoFrame,videoFrame,Size(0,0),2,2);//������Ƶͼ��Ĵ�С
//		src.copyTo(dst);
//		vector<Rect> found, found_filtered;//��������
//
//		//��ͼƬ���ж�߶Ȼ����˼�� 
//		cout<<"���ж�߶ȼ��"<<endl;
//		detectHOG.detectMultiScale(src, found, HitThreshold, WinStride, Size(0,0), DetScale, 2, false);
//		//������1Դͼ��2���������3���������ͳ�ƽ��ľ���4�ƶ�����(������block������������)5��Ե��չ6Դͼ��ͼ��ÿ����С����7�������8���෽ʽ
//
//		found_filtered = found;//����Ҫ����Ƕ�׾��ο� ��ֱ���ü��Ľ�����к�������
//		//for(int i=0; i < found.size(); i++)  //�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
//		//{  
//		//	Rect r = found[i];  
//		//	int j=0;  
//		//	for(; j < found.size(); j++)  
//		//		if(j != i && (r & found[j]) == r)  
//		//			break;  
//		//	if( j == found.size())  
//		//		found_filtered.push_back(r);  
//		//}
//		//cout<<"�ҵ��ľ��ο������"<<found_filtered.size()<<endl;
//		
//		//�Լ�⵽��ͼ����з���
//		for(int i=0; i<found_filtered.size(); i++)  
//		{
//			cout<<"width:"<<found_filtered[i].width<<"  height:"<<found_filtered[i].height<<endl;//�������ͼ��С
//			vector<float> descriptors;//HOG����������
//			Mat descriptorsMat = Mat::zeros(1, descriptorDim, CV_32FC1);//�����õ�HOG����������������=1������=��������ά��
//			Mat temp;
//			resize(src(found_filtered[i]),temp,WinSize);//���������ͼ��ߴ�
//
//			classifyHOG.compute(temp,descriptors);//����HOG������
//			for(int i=0; i<descriptorDim; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];//������������ֵ
//
//			float classifyResult = classifySvm.predict(descriptorsMat);//���������ͼ����Ԥ��
//
//			if (classifyResult == 1)//������
//			{
//				rectangle(dst, found_filtered[i], Scalar(255,0,0), 3);//��ͼ�л�������
//				if (SAVESET)//�Ƿ񱣴�������
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_1<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found_filtered[i]));
//				}
//			} 
//			else if (classifyResult == 2)//�ϰ���
//			{
//				rectangle(dst, found_filtered[i], Scalar(0,255,0), 3);//��ͼ�л�������
//				if (SAVESET)//�Ƿ񱣴�������
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_2<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found_filtered[i]));
//				}
//			}
//			else if (classifyResult ==3)//����
//			{
//				rectangle(dst, found_filtered[i], Scalar(0,0,255), 3);//��ͼ�л�������
//				if (SAVESET)//�Ƿ񱣴�������
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_3<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found_filtered[i]));
//				}
//			}
//			else//����
//			{
//				rectangle(dst, found_filtered[i], Scalar(255,255,255), 3);//��ͼ�л�������
//			}
//
//		}
//
//		//������Ƶͼ��
//		outputVideo<<dst;
//		imshow("dst",dst);
//		if(waitKey(1)>=0)stop = true;//ͨ������ֹͣ��Ƶ
//	}
//	//---------------------------------end---------------------------------------
//}  
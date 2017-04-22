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
//int main()  
//{
//	//svm,hog
//    MySVM detectSvm;
//	MySVM classifySvm;
//	detectSvm.load(DetectSvmName);
//	classifySvm.load(ClassifySvmName);
//	HOGDescriptor detectHOG(WinSizeDetect,BlockSizeDetect,BlockStrideDetect,CellSizeDetect,NbinsDetect,1,-1,0,0.2,false,10);
//	HOGDescriptor classifyHOG(WinSizeClassify,BlockSizeClassify,BlockStrideClassify,CellSizeClassify,NbinsClassify);//for calculating HOG
//	setHOG(detectSvm, detectHOG);//set detectHOG from detectSvm
//	int descriptorDimClassify = classifySvm.get_var_count();
//	//result files
//	const char * sd1 = {"md "ResultVideoFile_1};//for saving result image
//	system(sd1);
//	const char * sd2 = {"md "ResultVideoFile_2};
//	system(sd2);
//	const char * sd3 = {"md "ResultVideoFile_3};
//	system(sd3);
//	//video
//	VideoCapture myVideo(TestVideo);
//	if(!myVideo.isOpened()){cout<<"read video error"<<endl;system("puase");return -1;}
//	double videoRate=myVideo.get(CV_CAP_PROP_FPS);
//	int videoWidth=myVideo.get(CV_CAP_PROP_FRAME_WIDTH);
//	int videoHight=myVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
//	int videoDelay=1000/videoRate;
//	VideoWriter outputVideo(ResultVideo, CV_FOURCC('M', 'J', 'P', 'G'), videoRate, Size(videoWidth, videoHight));
//	//frame
//	Mat src,dst;
//	//label
//	LabelRobot label_irobot(50,10);
//	LabelRobot label_obstacle(50,4);
//	//begin
//	bool stop = false;
//	for (int fnum = 1;!stop;fnum++)
//	{
//		cout<<"frame: "<<fnum<<endl;
//		if (!myVideo.read(src)){cout<<"video end"<<endl;waitKey(0); break;}//get a frame
//		//resize(videoFrame,videoFrame,Size(0,0),2,2);
//		src.copyTo(dst);
//		vector<Rect> found;//detect result
//		cout<<"detect"<<endl;
//		detectHOG.detectMultiScale(src, found, HitThreshold, WinStride, Size(0,0), DetScale, 0.2, true);
//			//parameter: 1.src image 2.detect location result 3.(optional)detect weight result 4.offset of hyperplane 
//			//5. window stride (must be integer multiple to block stride!!!) 6.extend on src edge 7.scale resize ratio for every detection 
//			//8.cluster parameter (if meanshift cluster flag is false, parameter means minimum possible number of rectangles minus 1. if true, parameter's function is similar to HitThreshold)
//			//9.use meanshift cluster flag(false-cluster, still have overlap; true-meanshift cluster, without overlap)
//		//label
//		vector<RobotMessage> irobots_message;
//		vector<RobotMessage> obstacles_message;
//		//classify
//		for(int i=0; i<found.size(); i++)  
//		{
//			cout<<"width:"<<found[i].width<<"  height:"<<found[i].height<<endl;//detect windows
//			vector<float> descriptors;//classily HOG descriptor
//			Mat descriptorsMat = Mat::zeros(1, descriptorDimClassify, CV_32FC1);
//			Mat temp;
//			resize(src(found[i]),temp,WinSizeClassify);//resize detect result image for classify
//			classifyHOG.compute(temp,descriptors);//calculate classily HOG descriptor
//			for(int i=0; i<descriptorDimClassify; i++)  
//				descriptorsMat.at<float>(0,i) = descriptors[i];
//			float classifyResult = classifySvm.predict(descriptorsMat);//classily result
//			if (classifyResult == 1)//irobot
//			{
//				irobots_message.insert(irobots_message.end(),RobotMessage(found[i]));
//				rectangle(dst, found[i], Scalar(255,0,0), 3);
//				if (SAVESET)//save classify result
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_1<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found[i]));
//				}
//			} 
//			else if (classifyResult == 2)//obstacle
//			{
//				obstacles_message.insert(obstacles_message.end(),RobotMessage(found[i]));
//				rectangle(dst, found[i], Scalar(0,255,0), 3);
//				if (SAVESET)//save classify result
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_2<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found[i]));
//				}
//			}
//			else if (classifyResult ==3)//background
//			{
//				rectangle(dst, found[i], Scalar(0,0,255), 3);
//				if (SAVESET)//save classify result
//				{
//					strstream ss;
//					string s;
//					ss<<ResultVideoFile_3<<1000*fnum+i<<".jpg";
//					ss>>s;
//					imwrite(s,src(found[i]));
//				}
//			}
//		}
//		//label
//		label_irobot.input(irobots_message);
//		label_irobot.getLabel(irobots_message);
//		label_obstacle.input(obstacles_message);
//		label_obstacle.getLabel(obstacles_message);
//		for (int i = 0; i < irobots_message.size(); i++)
//		{
//			char label[8];
//			sprintf(label,"%d",irobots_message[i].label);
//			putText(dst,label,irobots_message[i].center,FONT_HERSHEY_SIMPLEX,2,CV_RGB(0,0,0),2,10);
//			//put word on image: image,word,pos,font,scale,color,thickness,line type
//		}
//		for (int i = 0; i < obstacles_message.size(); i++)
//		{
//			char label[8];
//			sprintf(label,"%d",obstacles_message[i].label);
//			putText(dst,label,obstacles_message[i].center,FONT_HERSHEY_SIMPLEX,2,CV_RGB(0,0,0),2,10);
//			//put word on image: image,word,pos,font,scale,color,thickness,line type
//		}
//		//save frame to video
//		outputVideo<<dst;
//		imshow("dst",dst);
//		if(waitKey(1)>=0)stop = true;
//	}
//}  
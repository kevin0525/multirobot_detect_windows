//-----------------------�궨��----------------------------
//---------------------------------------------------------
#define IrobotSetNo 430			//��������������  
#define ObstacleSetNo 239		//�ϰ���������
#define BackgroundSetNo 412		//������������
#define HardBackgroundSetNo 253	//Hard������������

#define SHOWSET false			//�Ƿ���ʾѵ������
#define TRAIN true				//�Ƿ����ѵ��,true��ʾѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
#define SAVESET true			//�Ƿ񱣴�������

/*//for cpu
//HOG�����Ӳ���
#define WinSizeDetect Size(32,16)		//��ⴰ�ڳߴ�
#define BlockSizeDetect Size(8,8)		//block�ߴ�
#define BlockStrideDetect Size(4,4)		//block����
#define CellSizeDetect Size(4,4)		//cell�ߴ�
#define NbinsDetect 9					//ֱ��ͼbin����

//HOG�����Ӳ���
#define WinSizeClassify Size(48,24)		//��ⴰ�ڳߴ�
#define BlockSizeClassify Size(16,8)		//block�ߴ�
#define BlockStrideClassify Size(4,4)	//block����
#define CellSizeClassify Size(4,4)		//cell�ߴ�
#define NbinsClassify 9					//ֱ��ͼbin����
*/

///////////for GPU
//HOG�����Ӳ���
#define WinSizeDetect Size(40,40)		//��ⴰ�ڳߴ�
#define BlockSizeDetect Size(16,16)		//block�ߴ�
#define BlockStrideDetect Size(8,8)		//block����
#define CellSizeDetect Size(8,8)		//cell�ߴ�
#define NbinsDetect 9					//ֱ��ͼbin����

//HOG�����Ӳ���
#define WinSizeClassify Size(40,40)		//��ⴰ�ڳߴ�
#define BlockSizeClassify Size(16,16)		//block�ߴ�
#define BlockStrideClassify Size(8,8)	//block����
#define CellSizeClassify Size(8,8)		//cell�ߴ�
#define NbinsClassify 9					//ֱ��ͼbin����

////HOG�����Ӳ���
//#define WinSize Size(20,10)		//��ⴰ�ڳߴ�
//#define BlockSize Size(4,4)		//block�ߴ�
//#define BlockStride Size(2,2)	//block����
//#define CellSize Size(2,2)		//cell�ߴ�
//#define Nbins 9					//ֱ��ͼbin����

////HOG�����Ӳ���
//#define WinSize Size(64,32)		//��ⴰ�ڳߴ�
//#define BlockSize Size(8,8)		//block�ߴ�
//#define BlockStride Size(4,4)		//block����
//#define CellSize Size(4,4)		//cell�ߴ�
//#define Nbins 9					//ֱ��ͼbin����

//detectMultiScale���ֲ���
#define HitThreshold 0			//���������볬ƽ����С����
#define WinStride Size(8,8)		//�ƶ�����(������block������������)
#define DetScale 1.1			//Դͼ��ͼ��ÿ����С����
#define DetectResizeRate 1.4	//boxes' resizerate of detect's output

#define TestImage "../Data/TestImage/13.jpg"				//���ڼ��Ĳ���ͼ��
#define ResultImage "../Data/Result/13.jpg"					//����ͼ��ļ����
#define ResultImageFile_1 "..\\Data\\Result\\13-1\\"		//����ͼ��ķ����ͼ�ļ���1
#define ResultImageFile_2 "..\\Data\\Result\\13-2\\"		//����ͼ��ķ����ͼ�ļ���2
#define ResultImageFile_3 "..\\Data\\Result\\13-3\\"		//����ͼ��ķ����ͼ�ļ���3
#define TestVideo "../Data/TestVideo/jade_s.avi"			//���ڼ��Ĳ�����Ƶ
#define ResultVideo "../Data/Result/jade_s.avi"			//������Ƶ�ļ����
#define ResultVideoFile_1 "..\\Data\\Result\\out_1\\"	//������Ƶ�ķ����ͼ�ļ���1
#define ResultVideoFile_2 "..\\Data\\Result\\out_2\\"	//������Ƶ�ķ����ͼ�ļ���2
#define ResultVideoFile_3 "..\\Data\\Result\\out_3\\"	//������Ƶ�ķ����ͼ�ļ���3

#define IrobotSetFile "../Data/IrobotSet/"					//����������ͼƬ�ļ���
#define ObstacleSetFile "../Data/ObstacleSet/"				//�ϰ�����ͼƬ�ļ���
#define BackgroundSetFile "../Data/BackgroundSet/"			//��������ͼƬ�ļ���
#define HardBackgroundSetFile "../Data/HardBackgroundSet/"	//Hard����ͼƬ�ļ���
#define SetName "0SetName.txt"								//����ͼƬ���ļ����б�txt
#define DetectSvmName "../Data/Result/SVM_HOG_Detect.xml"	//�������ȡ�ļ��ģ���ļ�����
#define ClassifySvmName "../Data/Result/SVM_HOG_Classify.xml"//�������ȡ�ķ���ģ���ļ�����


#define TrainPerc 1		//ѵ������
#define VaildPerc 0		//������֤����
//#define TestPerc 0.2
#define IrobotTrainNo (int(IrobotSetNo * TrainPerc))//ѵ����������
#define ObstacleTrainNo (int(ObstacleSetNo * TrainPerc))
#define BackgroundTrainNo (int(BackgroundSetNo * TrainPerc))
#define HardBackgroundTrainNo (int(HardBackgroundSetNo * TrainPerc))
#define AllTrainNo (IrobotTrainNo + ObstacleTrainNo + BackgroundTrainNo + HardBackgroundTrainNo)
#define IrobotVaildNo (int(IrobotSetNo * VaildPerc))//������֤��������
#define ObstacleVaildNo (int(ObstacleSetNo * VaildPerc))
#define BackgroundVaildNo (int(BackgroundSetNo * VaildPerc))
#define HardBackgroundVaildNo (int(HardBackgroundSetNo * VaildPerc))
#define AllVaildNo (IrobotVaildNo + ObstacleVaildNo + BackgroundVaildNo + HardBackgroundVaildNo)
#define IrobotTestNo (IrobotSetNo - IrobotTrainNo - IrobotVaildNo)//������������
#define ObstacleTestNo (ObstacleSetNo - ObstacleTrainNo - ObstacleVaildNo)
#define BackgroundTestNo (BackgroundSetNo - BackgroundTrainNo - BackgroundVaildNo)
#define HardBackgroundTestNo (HardBackgroundSetNo - HardBackgroundTrainNo - HardBackgroundVaildNo)
#define AllTestNo (IrobotSetNo + ObstacleSetNo + BackgroundSetNo + HardBackgroundSetNo)

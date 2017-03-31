#include <iostream>  
#include <fstream>  
#include <strstream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <time.h>

using namespace cv;
//-----------------------继承类----------------------------
//---------------------------------------------------------
//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，  
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问  
class MySVM : public CvSVM  
{  
public:  
	//获得SVM的决策函数中的alpha数组  
	double * get_alpha_vector()  
	{  
		return this->decision_func->alpha;  
	}  

	//获得SVM的决策函数中的rho参数,即偏移量  
	float get_rho()  
	{  
		return this->decision_func->rho;  
	}  
};  

//-------生成一个打乱的数组（从0开始的连续整数）-----------------
//---------------------------------------------------------
void random(int a[], int n)
{
	for (int nu = 0;nu<n;nu++)
	{
		a[nu] = nu;
	}
	int index, tmp, i;
	srand(time(NULL));
	for (i = 0; i <n; i++)
	{
		index = rand() % (n - i) + i;
		if (index != i)
		{
			tmp = a[i];
			a[i] = a[index];
			a[index] = tmp;
		}
	}
}

//--------根据元素大小赋予类型（0-train,1-vaild,2-test）----------------------
//---------------------------------------------------------
void typeHandle(int arr[],int setNo,int trainNo,int vaildNo)
{
	for (int i = 0;i<setNo;i++)
	{
		if (arr[i]<trainNo)
		{
			arr[i] = 0;
		} 
		else if(arr[i]<trainNo + vaildNo)
		{
			arr[i] = 1;
		}
		else
		{
			arr[i] = 2;
		}
	}
}

//struct for robot message
struct RobotMessage
{
	Rect location_image;//robot location on image
	Point2i center;//center of robot location on image
	int label;
	float distance_min;//minimum distance for point from this frame to last frame

	RobotMessage(){}

	RobotMessage(Rect location_image0)
	{
		location_image = location_image0;
		computerCenter();
		label = 0;
		distance_min = 10000;
	}

	~RobotMessage(){}

	void computerCenter()
	{
		center.x = location_image.x + location_image.width / 2;
		center.y = location_image.y + location_image.height / 2;
	}
};

//class for labeling robot
class LabelRobot
{
private:
	vector<RobotMessage> robots;
	vector<RobotMessage> robots_last;
	int label_max;//maximum used label
	int distance_max;//max distance from last position to this position
	int number_limit;//maximum robot number

public:
	LabelRobot()
	{
		label_max = 0;
		distance_max = 100000;
		number_limit = 10;
	}

	LabelRobot(int max_distance0, int number_limit0)
	{
		label_max = 0;
		distance_max = max_distance0;
		number_limit = number_limit0;
	}

	~LabelRobot(){}

	void input(const vector<RobotMessage> &input_irobots)
	{
		robots.clear();
		robots.insert(robots.end(), input_irobots.begin(), input_irobots.end());
	}

	void getLabel(vector<RobotMessage> &output_robots)
	{
		labelRobot();
		output_robots.clear();
		output_robots.insert(output_robots.end(), robots.begin(), robots.end());
		robots_last.clear();
		robots_last.insert(robots_last.end(), robots.begin(), robots.end());
	}

private:
	void labelRobot()
	{
		vector<RobotMessage> robots_temp;
		robots_temp.insert(robots_temp.end(), robots.begin(), robots.end());
		vector<RobotMessage> robots_last_temp;
		robots_last_temp.insert(robots_last_temp.end(), robots_last.begin(), robots_last.end());
		vector<RobotMessage> robots_labeled;

		while(1)
		{
			if (robots_temp.size() == 0 || robots_labeled.size() >= number_limit)
			{
				break;
			}

			if (robots_last_temp.size() == 0)
			{
				//addOtherLabel(robots_temp, robots_labeled);
				for (int i = 0; i<robots_temp.size();i++)
				{
					label_max++;
					robots_temp[i].label = label_max;
				}
				robots_labeled.insert(robots_labeled.end(), robots_temp.begin(), robots_temp.end());
				break;
			}

			float distance_min = 10000;
			int robot_number_min, pair_number_i, pair_number_min;
			for (int i = 0; i < robots_temp.size(); i++)
			{
				calculateMinDistance(robots_temp[i], robots_last_temp, pair_number_i);
				if (robots_temp[i].distance_min < distance_min)
				{
					robot_number_min = i;
					pair_number_min = pair_number_i;
					distance_min = robots_temp[i].distance_min;
				}
			}

			if (distance_min > distance_max)
			{
				//addOtherLabel(robots_temp, robots_labeled);
				for (int i = 0; i<robots_temp.size();i++)
				{
					label_max++;
					robots_temp[i].label = label_max;
				}
				robots_labeled.insert(robots_labeled.end(), robots_temp.begin(), robots_temp.end());
				break;
			} 
			else
			{
				robots_temp[robot_number_min].label = robots_last_temp[pair_number_min].label;
				robots_labeled.insert(robots_labeled.end(), robots_temp[robot_number_min]);
				robots_temp.erase(robots_temp.begin() + robot_number_min);
				robots_last_temp.erase(robots_last_temp.begin() + pair_number_min);
			}
		}

		robots.clear();
		robots.insert(robots.end(), robots_labeled.begin(), robots_labeled.end());
	}

	void calculateMinDistance(RobotMessage &robots_temp_i, vector<RobotMessage> &robots_last_temp, int &pair_number_i)
	{
		float dx, dy, distance;
		for (int i = 0; i<robots_last_temp.size(); i++)
		{
			dx = robots_temp_i.center.x - robots_last_temp[i].center.x;
			dy = robots_temp_i.center.y - robots_last_temp[i].center.y;
			distance = sqrt(dx*dx+dy*dy);
			if (distance <= robots_temp_i.distance_min)
			{
				robots_temp_i.distance_min = distance;
				pair_number_i = i;
			}
		}
	}

	//void addOtherLabel(vector<RobotMessage> &robots_temp, vector<RobotMessage> &robots_labeled)
	//{
	//	for (int i = 0; i<robots_temp.size();i++)
	//	{
	//		label_max++;
	//		robots_temp[i].label = label_max;
	//	}
	//	robots_labeled.insert(robots_labeled.end(), robots_temp.begin(), robots_temp.end());
	//}
};


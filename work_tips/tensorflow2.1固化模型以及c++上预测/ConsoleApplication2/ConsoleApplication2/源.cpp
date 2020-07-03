#include<iostream>
#include <Python.h>
#include<windows.h>
#include <io.h>
#include <string>
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <numpy/arrayobject.h>
#include <object.h>
#include<ctime>
using namespace std;




int comp(string& a, string& b)
{
	if (a.length() == b.length())
	{
		return a<b;
	}
	if (a.length() < b.length())
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = long long(_findfirst(p.assign(path).append("\\*").c_str(), &fileinfo))) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				//                                 if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
				//                                         getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	std::sort(files.begin(), files.end(), &comp);
}

void testImage(char *path)
{
	try{
		Py_SetPythonHome(L"C:\\Anaconda3\\");//这里地址一定要写对啊！
		Py_Initialize();
		import_array(); // 用于numpy
		PyEval_InitThreads();

		PyObject*pFunc = NULL;
		PyObject*pArg = NULL;
		PyObject* pModel = NULL;

		clock_t startTime, endTime;  // 用于计算每张图片耗时
		startTime = clock();
		pModel = PyImport_ImportModule("pb_test");
		//导入模块名字，python脚本文件名为mnistPre（注意路径，不写路径时，需要将脚本拷贝到程序同一路径）
		PyObject *pDict = PyModule_GetDict(pModel);
		//获取模块中的内容
		PyObject *pClass_mnist = PyDict_GetItemString(pDict, "Classify");//查找类
		//查找获得的模块中的内容
		PyObject *pIns_mnist = PyInstanceMethod_New(pClass_mnist);//实例化类对象
		//实例化一个查找到的pClass_mnist类构造对象
		PyObject *pInstance = PyObject_CallObject(pIns_mnist, nullptr);//实例化对象
		//类对象实例化
		PyObject* reasult = PyObject_CallMethod(pInstance, "recognize", "s", path);
		//类方法调用
		endTime = clock();
		cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;


		vector<string> m_vsFiles;
		getFiles("D:\\data\\test_2", m_vsFiles);
		vector<string>::iterator iter;
		for (iter = m_vsFiles.begin(); iter != m_vsFiles.end(); iter++)
		{
			string strFileName = *iter;
			int nDotIndex = int(strFileName.find_last_of('.'));
			bool bIsBmp = false;
			if (0 <= nDotIndex && nDotIndex < strFileName.length())
			{
				string strPostfix = strFileName.substr(nDotIndex + 1, strFileName.length() - nDotIndex - 1);
				if (strPostfix == "bmp")
				{
					bIsBmp = true;
				}
			}
			if (false == bIsBmp)
			{
				continue;
			}
			
			cv::Mat img = cv::imread(iter->c_str(), 1);
			//m_readImg.Init(img.cols, img.rows, img.channels())
			//PyArray_SimpleNewFromData();
			npy_intp Dims[3] = { img.rows, img.cols, img.channels() }; //给定维度信息
			PyObject*PyArray = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, img.data);

			PyObject_CallMethod(pInstance, "recognize", "O", PyArray);
		}


		//for (int i = 0; i < temp.size(); i++){
		//	cout << temp[i] << endl;
		//	cv::Mat img = cv::imread(temp[i], cv::IMREAD_REDUCED_COLOR_8);
		//    // PyObject_CallMethod(pInstance, "recognize", "s", temp[i]);
		//}	
	}
	catch (exception& e)
	{
		cout << "Standard exception: " << e.what() << endl;
	}
}
int main()
{
	//cv::imshow("BGR", img);
	//cv::waitKey(0);
	char * path = "D:\\classification_demo\\two_kinds_model\\freeze\\45.bmp";
	testImage(path);
	system("pause");
	return 0;
}

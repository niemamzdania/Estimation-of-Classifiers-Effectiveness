#include "pch.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <cstdlib>

using namespace std;

void Read(double** data, string name);

vector <vector<int> > Save_classes(double** data, const int columns, const int rows);

vector <double> Bayes(double** data, double** data2, const int columns, const int rows, const int rows2);

void Train_and_Test(const int columns, const int rows_original);

void Monte_Carlo_Cross_Validation(const int columns, const int rows_original);

void Cross_Validation(const int columns, const int rows_original);

void Leave_one_out(const int columns, const int rows_original);

void Bagging(const int columns, const int rows_original);

int main()
{
	const int columns = 15;
	const int rows_original = 690;

	cout << "Naive Bayes Classifier for methods:" << endl << endl;

	Train_and_Test(columns, rows_original);

	Monte_Carlo_Cross_Validation(columns, rows_original);

	Cross_Validation(columns, rows_original);

	Leave_one_out(columns, rows_original);

	Bagging(columns, rows_original);
}

void Read(double** data, string name)
{
	fstream file;
	file.open(name, ios::in);

	if (file.good())
	{
		string line; //line from file
		string digit; //single value from file

		int number_of_row = 0;

		while (!file.eof())
		{
			getline(file, line);

			int counter = 0;

			for (int i = 0; i <= line.length(); i++)
			{
				if (line[i] == ' ' || line[i] == '\0')
				{
					double number = stod(digit);

					data[counter][number_of_row] = number;

					digit = "";
					counter++;
				}
				else
				{
					digit += line[i];
				}
			}

			number_of_row++;
		}
	}
	else
	{
		cout << "File error. Check the catalog with the project." << endl;
		system("PAUSE");
		exit(0);
	}

	file.close();
}

vector <vector<int> > Save_classes(double** data, const int columns, const int rows)
{
	vector< vector<int> > classes;

	bool flag = false;
	int counter = 0;
	for (int i = 0; i < rows; i++)
	{
		flag = false;
		for (int j = 0; j < classes.size(); j++)
		{
			if (classes[j][0] == (int)data[columns - 1][i])
				flag = true;
		}

		if (flag == false)
		{
			counter = 0;
			for (int j = 0; j < rows; j++)
			{
				if (data[columns - 1][j] == data[columns - 1][i])
					counter++;
			}
			classes.push_back({ (int)data[columns - 1][i], counter });
		}
	}

	sort(classes.begin(), classes.end());

	return classes;
}

vector <double> Bayes(double** data, double** data2, const int columns, const int rows, const int rows2)
{
	//cout << "Naive Bayes classifier " << "for method " << name_of_method << ":" <<  endl;
	//Save available classes into vector
	vector <vector<int> > classes = Save_classes(data2, columns, rows2); //{class, number of occurrences}
	vector <vector<int> > classes_tst = Save_classes(data, columns, rows);

	//Determinig the data
	int counter;
	double sum;
	double class_size;
	vector<double> param_c;
	int decision = -1;
	vector <vector <int> > classification; //classification for objects { decision(from system), true/false classification }

	srand(time(NULL));

	for (int i = 0; i < rows; i++)
	{
		param_c.clear();
		for (int j = 0; j < classes.size(); j++)
		{
			class_size = (double)classes[j][1] / (double)rows2;
			sum = 0;

			for (int k = 0; k < columns - 1; k++)
			{
				counter = 0;
				for (int l = 0; l < rows2; l++)
				{
					if (data2[k][l] == data[k][i] && (int)data2[columns - 1][l] == classes[j][0])
						counter++;
				}
				sum += (double)counter / (double)classes[j][1];
			}
			
			param_c.push_back(sum * class_size);
		}
		
		//counting param_c is correct
		if (param_c[0] > param_c[1])
			decision = classes[0][0];
		else if (param_c[0] < param_c[1])
			decision = classes[1][0];
		else
		{
			int number_of_index = rand() % classes.size();
			decision = classes[number_of_index][0];
		}

		if (decision == data[columns - 1][i])
			classification.push_back({decision, 1});
		else classification.push_back({ decision, 0 });
	}

	int number_of_true_classificated_objects = 0;

	for (int i = 0; i < classification.size(); i++)
	{
		if (classification[i][1] == 1)
			number_of_true_classificated_objects++;

		//cout << classification[i][0] << " " << classification[i][1] << endl;
	}

	double global_accuracy = (double)number_of_true_classificated_objects / (double)classification.size();

	int number_of_classificated_objects;
	double sum_accuracy = 0;

	for (int i = 0; i < classes.size(); i++)
	{
		number_of_true_classificated_objects = 0;
		number_of_classificated_objects = 0;

		for (int j = 0; j < classification.size(); j++)
		{
			if (classification[j][0] == classes[i][0] && classification[j][1] == 1)
				number_of_true_classificated_objects++;
			if (data[columns-1][j] == classes[i][0])
				number_of_classificated_objects++;
		}
		
		if (number_of_classificated_objects == 0)
			sum_accuracy += 0;
		else sum_accuracy += (double)number_of_true_classificated_objects / (double)number_of_classificated_objects;
	}

	double balanced_accuracy = sum_accuracy / (double)classes_tst.size();

	//cout << endl << "Global accuracy: " << global_accuracy << endl << endl;
	//cout << "Balanced accuracy: " << balanced_accuracy << endl;

	vector <double> accuracy;

	accuracy.push_back(global_accuracy);
	accuracy.push_back(balanced_accuracy);

	return accuracy;
}

void Train_and_Test(const int columns, const int rows_original)
{
	cout << "Train and Test:" << endl << endl;

	//Read data from files
	const float ratio = 0.5;
	const int rows = rows_original * ratio;
	const int rows2 = rows_original * (1 - ratio);

	double ** data;
	data = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data[i] = new double[rows_original];
	}

	Read(data, "australian.txt");
	
	double **data2;

	data2 = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data2[i] = new double[rows];
	}

	double **data3;

	data3 = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data3[i] = new double[rows2];
	}

	vector<int> numbers_of_objects;
	int number_of_object = 0;
	bool flag = false; //false - object is correct

	//Drawing lots of objects
	srand(time(NULL));
	while (numbers_of_objects.size() != rows)
	{
		number_of_object = rand() % rows_original;

		flag = false;
		for (int i = 0; i < numbers_of_objects.size(); i++)
		{
			if (numbers_of_objects[i] == number_of_object)
				flag = true;
		}
		if (flag == false)
			numbers_of_objects.push_back(number_of_object);
	}

	for (int i = 0; i < numbers_of_objects.size(); i++)
	{
		for (int j = 0; j < columns; j++)
		{
			data3[j][i] = data[j][numbers_of_objects[i]];
		}
	}
	
	int counter = 0;

	for (int i = 0; i < rows_original; i++)
	{
		flag = false;
		for (int j = 0; j < numbers_of_objects.size(); j++)
		{
			if (i == numbers_of_objects[j])
				flag = true;
		}
		if (flag == false)
		{
			for (int j = 0; j < columns; j++)
			{
				data2[j][counter] = data[j][i];
			}
			counter++;
		}
	}

	//Calling the algorithm
	vector <double> accuracy = Bayes(data2, data3, columns, rows, rows2);

	cout << "Global accuracy: " << accuracy[0] << endl;
	cout << "Balanced accuracy: " << accuracy[1] << endl;

	//Memory release
	for (int i = 0; i < columns; i++)
	{
		delete[] data[i];
		delete[] data2[i];
		delete[] data3[i];
	}
	delete[] data;
	delete[] data2;
	delete[] data3;
}

void Monte_Carlo_Cross_Validation(const int columns, const int rows_original)
{
	cout << endl << endl << "Monte Carlo Cross Validation:" << endl << endl;

	int number_of_tests = 5;

	//Read data from files
	const float ratio = 0.5;
	const int rows = rows_original * ratio;
	const int rows2 = rows_original * (1 - ratio);

	double ** data;
	data = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data[i] = new double[rows_original];
	}

	Read(data, "australian.txt");

	double **data2;

	data2 = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data2[i] = new double[rows];
	}

	double **data3;

	data3 = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data3[i] = new double[rows2];
	}

	vector<int> numbers_of_objects;
	int number_of_object = 0;
	bool flag = false; //false - object is correct

	vector < vector<double> > accuracy;

	srand(time(NULL));

	for (int a = 0; a < number_of_tests; a++)
	{
		numbers_of_objects.clear();

		//Drawing lots of objects
		while (numbers_of_objects.size() != rows2)
		{
			number_of_object = rand() % rows_original;

			flag = false;
			for (int i = 0; i < numbers_of_objects.size(); i++)
			{
				if (numbers_of_objects[i] == number_of_object)
					flag = true;
			}
			if (flag == false)
				numbers_of_objects.push_back(number_of_object);
		}

		for (int i = 0; i < numbers_of_objects.size(); i++)
		{
			for (int j = 0; j < columns; j++)
			{
				data3[j][i] = data[j][numbers_of_objects[i]];
			}
		}

		int counter = 0;

		for (int i = 0; i < rows_original; i++)
		{
			flag = false;
			for (int j = 0; j < numbers_of_objects.size(); j++)
			{
				if (i == numbers_of_objects[j])
					flag = true;
			}
			if (flag == false)
			{
				for (int j = 0; j < columns; j++)
				{
					data2[j][counter] = data[j][i];
				}
				counter++;
			}
		}

		//Calling the algorithm
		accuracy.push_back(Bayes(data2, data3, columns, rows, rows2));
	}

	double sum_global_accuracy = 0;
	double sum_balanced_accuracy = 0;

	for (int i = 0; i < accuracy.size(); i++)
	{
		sum_global_accuracy += accuracy[i][0];
		sum_balanced_accuracy += accuracy[i][1];
	}

	sum_global_accuracy /= number_of_tests;
	sum_balanced_accuracy /= number_of_tests;

	cout << "Global accuracy: " << sum_global_accuracy << endl;
	cout << "Balanced accuracy: " << sum_balanced_accuracy << endl;

	//Memory release
	for (int i = 0; i < columns; i++)
	{
		delete[] data[i];
		delete[] data2[i];
		delete[] data3[i];
	}
	delete[] data;
	delete[] data2;
	delete[] data3;
}

void Cross_Validation(const int columns, const int rows_original)
{
	cout << endl << endl << "Cross Validation:" << endl << endl;

	int number_of_tests = 5;

	//Read data from files
	const int rows = rows_original / number_of_tests;
	const int rows2 = rows_original - (rows_original / number_of_tests);

	double ** data;
	data = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data[i] = new double[rows_original];
	}

	Read(data, "australian.txt");

	double **data2;

	data2 = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data2[i] = new double[rows];
	}

	double **data3;

	data3 = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data3[i] = new double[rows2];
	}

	vector<int> numbers_of_objects;
	vector<int> numbers_of_objects_in_one_test;
	int number_of_object = 0;
	bool flag = false; //false - object is correct

	vector < vector<double> > accuracy;

	srand(time(NULL));

	for (int a = 0; a < number_of_tests; a++)
	{
		numbers_of_objects_in_one_test.clear();

		//Drawing lots of objects
		while (numbers_of_objects_in_one_test.size() != rows)
		{
			number_of_object = rand() % rows_original;

			flag = false;
			for (int i = 0; i < numbers_of_objects.size(); i++)
			{
				if (numbers_of_objects[i] == number_of_object)
					flag = true;
			}
			if (flag == false)
			{
				numbers_of_objects_in_one_test.push_back(number_of_object);
				numbers_of_objects.push_back(number_of_object);
			}
		}

		for (int i = 0; i < numbers_of_objects_in_one_test.size(); i++)
		{
			for (int j = 0; j < columns; j++)
			{
				data2[j][i] = data[j][numbers_of_objects_in_one_test[i]];
			}
		}

		int counter = 0;

		for (int i = 0; i < rows_original; i++)
		{
			flag = false;
			for (int j = 0; j < numbers_of_objects_in_one_test.size(); j++)
			{
				if (i == numbers_of_objects_in_one_test[j])
					flag = true;
			}
			if (flag == false)
			{
				for (int j = 0; j < columns; j++)
				{
					data3[j][counter] = data[j][i];
				}
				counter++;
			}
		}
		
		//Calling the algorithm
		accuracy.push_back(Bayes(data2, data3, columns, rows, rows2));
	}

	double sum_global_accuracy = 0;
	double sum_balanced_accuracy = 0;

	for (int i = 0; i < accuracy.size(); i++)
	{
		sum_global_accuracy += accuracy[i][0];
		sum_balanced_accuracy += accuracy[i][1];
	}

	sum_global_accuracy /= number_of_tests;
	sum_balanced_accuracy /= number_of_tests;

	cout << "Global accuracy: " << sum_global_accuracy << endl;
	cout << "Balanced accuracy: " << sum_balanced_accuracy << endl;

	//Memory release
	for (int i = 0; i < columns; i++)
	{
		delete[] data[i];
		delete[] data2[i];
		delete[] data3[i];
	}
	delete[] data;
	delete[] data2;
	delete[] data3;
}

void Leave_one_out(const int columns, const int rows_original)
{
	cout << endl << endl << "Leave one out:" << endl << endl;

	int number_of_tests = rows_original;

	//Read data from files
	const int rows = rows_original / number_of_tests;
	const int rows2 = rows_original - (rows_original / number_of_tests);

	double ** data;
	data = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data[i] = new double[rows_original];
	}

	Read(data, "australian.txt");

	double **data2;

	data2 = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data2[i] = new double[rows];
	}

	double **data3;

	data3 = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data3[i] = new double[rows2];
	}

	vector<int> numbers_of_objects;
	vector<int> numbers_of_objects_in_one_test;
	int number_of_object = 0;

	vector < vector<double> > accuracy;

	for (int a = 0; a < number_of_tests; a++)
	{
		numbers_of_objects.clear();
		number_of_object = a;

		for (int i = 0; i < rows_original; i++)
		{
			if (i == number_of_object)
				continue;
			else numbers_of_objects.push_back(i);
		}

		for (int i = 0; i < columns; i++)
		{
			data2[i][0] = data[i][a];
		}

		for (int i = 0; i < numbers_of_objects.size(); i++)
		{
			for (int j = 0; j < columns; j++)
			{
				data3[j][i] = data[j][numbers_of_objects[i]];
			}
		}

		//Calling the algorithm
		accuracy.push_back(Bayes(data2, data3, columns, rows, rows2));
	}

	double sum_global_accuracy = 0;
	double sum_balanced_accuracy = 0;

	for (int i = 0; i < accuracy.size(); i++)
	{
		sum_global_accuracy += accuracy[i][0];
		sum_balanced_accuracy += accuracy[i][1];
	}

	sum_global_accuracy /= number_of_tests;
	sum_balanced_accuracy /= number_of_tests;

	cout << "Global accuracy: " << sum_global_accuracy << endl;
	cout << "Balanced accuracy: " << sum_balanced_accuracy << endl;

	//Memory release
	for (int i = 0; i < columns; i++)
	{
		delete[] data[i];
		delete[] data2[i];
		delete[] data3[i];
	}
	delete[] data;
	delete[] data2;
	delete[] data3;
}

void Bagging(const int columns, const int rows_original)
{
	cout << endl << endl << "Bagging:" << endl << endl;

	int number_of_tests = 5;
	int number_of_draws = rows_original;

	//Read data from files
	double ** data;
	data = new double*[columns];

	for (int i = 0; i < columns; i++)
	{
		data[i] = new double[rows_original];
	}

	Read(data, "australian.txt");

	//drawing of test system objects
	srand(time(NULL));
	int number_of_object = 0;
	vector <int> numbers_of_objects;
	bool flag = false; //false - object is correct
	vector< vector<double> > accuracy;
	vector <int> numbers_of_test_objects;

	for (int a = 0; a < number_of_tests; a++)
	{
		numbers_of_test_objects.clear();
		numbers_of_objects.clear();
		

		for (int i = 0; i < number_of_draws; i++)
		{
			number_of_object = rand() % 690;

			flag = false;
			for (int j = 0; j < numbers_of_objects.size(); j++)
			{
				if (number_of_object == numbers_of_objects[j])
					flag = true;
			}
			if (flag == false)
				numbers_of_objects.push_back(number_of_object);
		}

		const int rows = rows_original - numbers_of_objects.size();
		const int rows2 = numbers_of_objects.size();

		double **data2;

		data2 = new double*[columns];

		for (int i = 0; i < columns; i++)
		{
			data2[i] = new double[rows];
		}

		double **data3;

		data3 = new double*[columns];

		for (int i = 0; i < columns; i++)
		{
			data3[i] = new double[rows2];
		}

		for (int i = 0; i < numbers_of_objects.size(); i++)
		{
			for (int j = 0; j < columns; j++)
			{
				data3[j][i] = data[j][numbers_of_objects[i]];
			}
		}

		for (int i = 0; i < rows_original; i++)
		{
			flag = false;
			for (int j = 0; j < numbers_of_objects.size(); j++)
			{
				if (i == numbers_of_objects[j])
					flag = true;
			}
			if (flag == false)
			{
				numbers_of_test_objects.push_back(i);
			}
		}

		for (int i = 0; i < numbers_of_test_objects.size(); i++)
		{
			for (int j = 0; j < columns; j++)
			{
				data2[j][i] = data[j][numbers_of_test_objects[i]];
			}
		}

		accuracy.push_back(Bayes(data2, data3, columns, rows, rows2));

		for (int i = 0; i < columns; i++)
		{
			delete[] data2[i];
			delete[] data3[i];
		}
		delete[] data2;
		delete[] data3;
	}

	double sum_global_accuracy = 0;
	double sum_balanced_accuracy = 0;

	for (int i = 0; i < accuracy.size(); i++)
	{
		sum_global_accuracy += accuracy[i][0];
		sum_balanced_accuracy += accuracy[i][1];
	}

	sum_global_accuracy /= number_of_tests;
	sum_balanced_accuracy /= number_of_tests;

	cout << "Global accuracy: " << sum_global_accuracy << endl;
	cout << "Balanced accuracy: " << sum_balanced_accuracy << endl;

	//Memory release
	for (int i = 0; i < columns; i++)
	{
		delete[] data[i];
	}
	delete[] data;
}
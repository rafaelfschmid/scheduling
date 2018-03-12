// basic file operations
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>

using namespace std;

#ifndef REPEAT
#define REPEAT 20
#endif

const vector<string> explode(const string& s, const char& c)
{
	string buff{""};
	vector<string> v;

	for(auto n:s)
	{
		if(n != c) buff+=n; else
		if(n == c && buff != "") { v.push_back(buff); buff = ""; }
	}
	if(buff != "") v.push_back(buff);

	return v;
}

int main(int argc, char **argv) {

	if (argc < 3) {
		printf("Parameters missing: <file input> <file output>\n\n");
		return 0;
	}

	std::vector<std::vector<string> > matrix;

	ofstream output(argv[1]);
	printf("%s\n", argv[1]);

	string param = argv[2];
	vector<string> files{explode(param, '\n')};

	ifstream input(files[0]);
	std::cout << files[0];

	string line;
	if (input.is_open()) {

		std::vector<string> aux; //the name of the files
		aux.push_back(" ");
		matrix.push_back(aux);

		while (getline(input, line)) {
			std::vector<string> times;
			times.push_back(line);

			double mean = 0;
			for (int i = 0; i < REPEAT; i++) {
				getline(input, line);
			}

			matrix.push_back(times);

			getline(input, line);
		}
		input.close();

	}

	for(auto f:files) {
		ifstream input(f);
		std::cout << f << "\n";

		string line;
		if (input.is_open()) {
			vector<string> name{explode(f, '/')};

			matrix[0].push_back(name[name.size()-1]);
			int k = 1;
			while (getline(input, line)) {
				std::vector<string> times;

				double mean = 0;
				for (int i = 0; i < REPEAT; i++) {
					getline(input, line);

					double value = stod(line);
					mean += value;
				}
				mean /= REPEAT;
				matrix[k].push_back(std::to_string(mean));

				k++;
				getline(input, line);
			}
			input.close();

		}
	}

	for(int i = 0; i < matrix[0].size(); i++) {
		for (int j = 0; j < matrix.size(); j++) {
			output << std::fixed << matrix[j][i] << "\t";
		}
		output << "\n";
	}

	output.close();

	return 0;
}

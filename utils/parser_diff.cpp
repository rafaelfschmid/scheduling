// basic file operations
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>
using namespace std;

int main(int argc, char **argv) {

	std::vector<std::vector<std::vector<double> > > matrix;
	std::vector<double> segments;
	ifstream input(argv[1]);
	ofstream output(argv[2]);

	string line;
	if (input.is_open()) {
		while (getline(input, line)) {
			std::vector<std::vector<double> > multiple_times;
			int seg = stoi(line);
//			cout << seg << "\n";
			std::vector<double> segment;
			for (int i = 0; i < 11; i++) {
				segment.push_back(seg);
			}
			multiple_times.push_back(segment);

			while (true) {
				getline(input, line);
				int size = stoi(line);
				std::vector<double> times;
				times.push_back(size);
//				cout << size << "\n";
				for (int i = 0; i < 10; i++) {
					getline(input, line);
//					cout << line << "\n";
					times.push_back(stod(line));
				}
				multiple_times.push_back(times);

				getline(input, line);

				if (size >= 1048576) {
					break;
				}

				getline(input, line);
			}
			matrix.push_back(multiple_times);
		}
		input.close();

		for (int k = 0; k < 11; k++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				output << matrix[0][j][k] << "\t";
//				cout << matrix[0][j][k] << ";";
			}
			output << "\n";
//			cout << "\n";
		}

		for (int i = 1; i < matrix.size(); i++) {
			for (int k = 1; k < 11; k++) {
				for (int j = 0; j < matrix[i].size(); j++) {
					output << matrix[i][j][k] << "\t";
//					cout << matrix[i][j][k] << ";";
				}
				output << "\n";
//				cout << "\n";
			}
		}
		output.close();
	}

	return 0;
}

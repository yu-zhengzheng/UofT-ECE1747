#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <sstream>
#include <thread>
#include "math.h"
#include <cstdio>
#include <mpi.h>
#include <deque>
#include <mutex>
#include <chrono>

using namespace std;

deque<int> range; //range of indices of particles
mutex mtx;  //mutex to protect range

vector<int*> read_csv(const string& file_name,int particles) {
    // Create a vector to store the int arrays
    vector<int*> result;
    // Create an input file stream object and open the file
    ifstream fin;
    fin.open(file_name);
    // Check if the file is opened successfully
    if (fin.is_open()){
        // Create a string to store each line of the file
        string line;
        while (std::getline(fin, line)&&particles>0) {
            particles--;//only read particles lines
            int tmp=0;
            int index=0;
            int* arr = new int[3];
            for (char c : line) {
                if (c == 'p') {
                    arr[index] = 1;
                } else if (c == 'e') {
                    arr[index] = -1;
                } else if (c==','){
                    arr[index]=tmp;
                    //cout<<tmp<<endl;
                    tmp=0;
                    index++;
                } else {
                    tmp=tmp*10+c-'0';
                }
            }
            result.push_back(arr);
        }
        // Close the file
        fin.close();
    } else {
        // If the file is not opened, print an error message
        cout << "Error: Unable to open file " << file_name << endl;
    }
    return result;
}

double find_min_force(vector<int*> vec, int* p1){
    int ind=0;
    double r=0;//min distance

    for (int j = 0; j < vec.size(); j++) {
        // Get the pointer to the current row
        int *p2 = vec[j];
        //cout << p1[0] << "," << p1[1] << " " << p2[0] << ","<< p2[1] << endl;
        double r_curr = pow(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2),0.5);
        if (r_curr<r && r_curr>0 || r==0){
            ind=j;
            r=r_curr;
        }
    }
    int *p2 = vec[ind];
    //return -force*charge of p1* charge of p2
    return -2.30144e-28/r/r * p1[2] * p2[2];
}

vector<double> compute(vector<int*> vec, vector<int> range){
    //thread function for mode 2
    vector<double> result;
    for (int i : range) {
        int *p=vec[i];
        result.push_back(find_min_force(vec,p));
        //cout<<i<<endl;
    }
    return result;
}

vector<double> compute_chunk(vector<int*> vec){
//vector<double> compute_chunk(vector<int*> vec, int i){//use this line to check if work is done by multiple threads
    //thread function for mode 3
    vector<double> result;
    while(true){
        std::lock_guard<std::mutex> lock(mtx); // Lock the queue
        if (!range.empty()) {
            int index = range.front();
            range.pop_front();
            int *p=vec[index];
            result.push_back(find_min_force(vec,p));
            //cout<<i<<"th thread: "<<index<<endl;
            //cout<<index<<"th force: "<<find_min_force(vec,p)<<endl;
        } else {
            // Queue is empty, thread exits
            break;
        }
    }
    return result;
}

void write_csv(vector<double> vec){
    ofstream csv_file("verification_results.csv");
    if (csv_file.is_open()){
        // Loop over the array elements
        for (int i = 0; i < 100; i++)
        {
            // Write the element to the file
            csv_file << vec[i]<<endl;
        }
        csv_file.close();
    }else{
        cout << "Unable to open file." << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4){
        std::cerr << "Usage: " << argv[0] << " <mode> <threads> <particles>" << std::endl;
        return 99;
    }

    int mode = std::stoi(argv[1]);
    int threads = std::stoi(argv[2]);
    int particles = std::stoi(argv[3]);

    auto start = chrono::system_clock::now();
    auto c1 = chrono::system_clock::now();
    vector<int*> data = read_csv("particles-student-1.csv",particles);
    auto c2 = chrono::system_clock::now();
    chrono::duration<double> elapsed = (c2 - start)*1000;
    cout<<"reading time: "<<elapsed.count()<<"ms"<<endl;

    vector<double> result;
    if (mode==1) {
        for (int i = 0; i < data.size(); i++) {
            int *p1 = data[i];
            double force = find_min_force(data, p1);
            result.push_back(force);
        }
        c1 = chrono::system_clock::now();
        elapsed = (c1 - c2)*1000;
        cout<<"computation time: "<<elapsed.count()<<"ms"<<endl;
    }else if (mode==2){
        double segmentSize = static_cast<double>(data.size()) / threads;
        vector<thread> threads_vec;
        for (int i = 0; i < threads; i++){
            int start = static_cast<int>(i * segmentSize);
            int end = static_cast<int>((i + 1) * segmentSize);
            if (i == threads - 1) {
                // Last thread should take any remaining values
                end = data.size();
            }
            vector<int> range;
            for (int j = start; j < end; j++) {
                range.push_back(j);
            }
            threads_vec.push_back(thread(compute,data,range));
        }
        c1 = chrono::system_clock::now();
        elapsed = (c1 - c2)*1000;
        cout<<"data partition and thread creation time: "<<elapsed.count()<<"ms"<<endl;

        for(int i = 0; i < threads; i++) {
            threads_vec[i].join();
        }
        c2 = chrono::system_clock::now();
        elapsed = (c2 - c1)*1000;
        cout<<"computation time: "<<elapsed.count()<<"ms"<<endl;
    }else if (mode==3){
        MPI_Init(0, 0);
        int rank;
        int processes;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        //cout<<"the rank of this process is: "<<rank<<endl;
        double segmentSize = static_cast<double>(data.size()) / processes;
        vector<thread> threads_vec;
        int start = static_cast<int>(rank * segmentSize);
        int end =min(static_cast<int>((rank + 1) * segmentSize),static_cast<int>(data.size()));
        for (int j = start; j < end; j++) {
            range.push_back(j);
        }
        c1 = chrono::system_clock::now();
        elapsed = (c1 - c2)*1000;
        cout<<"data partition time: "<<elapsed.count()<<"ms"<<endl;

        //initialize threads
        for (int i = 0; i < threads; i++){
            threads_vec.push_back(thread(compute_chunk,data));
            //threads_vec.push_back(thread(compute_chunk,data,i)); //use this line to check if work is done by multiple threads
        }
        c2 = chrono::system_clock::now();
        elapsed = (c2 - c1)*1000;
        cout<<"thread creation time: "<<elapsed.count()<<"ms"<<endl;

        for(int i = 0; i < threads; i++) {
            threads_vec[i].join();
        }
        //cout<<"process "<<rank<<" finished"<<endl;
        MPI_Finalize();
        c1 = chrono::system_clock::now();
        elapsed = (c1 - c2)*1000;
        cout<<"computation time: "<<elapsed.count()<<"ms"<<endl;
    }else{
        cout<<"unsupported mode!"<<endl;
        return -4;
    }

    if (mode==1) {
        write_csv(result);
    }
    auto end = std::chrono::system_clock::now();
    elapsed = (end - start)*1000;
    cout << "total execution time: " << elapsed.count() << "ms" << endl;
    return 0;
}

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

using namespace nvinfer1;
using namespace std;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

ICudaEngine* loadEngine(const string& engineFile, IRuntime* runtime) {
    ifstream file(engineFile, ios::binary);
    if (!file.good()) {
        cerr << "cannot open engine file: " << engineFile << endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char* data = new char[size];
    file.read(data, size);
    file.close();

    ICudaEngine* engine = runtime->deserializeCudaEngine(data, size, nullptr);
    delete[] data;
    return engine;
}

float* preprocessImage(const string& imagePath, int& inputH, int& inputW) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        cerr << "cannot open image: " << imagePath << endl;
        return nullptr;
    }

    // resize
    inputH = 224;
    inputW = 224;
    cv::resize(img, img, cv::Size(inputW, inputH));
    // normal
    img.convertTo(img, CV_32FC3, 1.0 / 255);

    // stand
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < inputH * inputW; ++i) {
            img.at<cv::Vec3f>(i)[c] = (img.at<cv::Vec3f>(i)[c] - mean[c]) / std[c];
        }
    }

    // CHW transform
    float* input = new float[3 * inputH * inputW];
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < inputH * inputW; ++i) {
            input[c * inputH * inputW + i] = img.at<cv::Vec3f>(i)[c];
        }
    }
    return input;
}

vector<float> softmax(const float* input, int length) {
    vector<float> probabilities(length);
    // get max
    float max_val = input[0];
    for(int i = 1; i < length; ++i) {
        if(input[i] > max_val)
            max_val = input[i];
    }

    // log sum
    float sum = 0.0f;
    for(int i = 0; i < length; ++i) {
        probabilities[i] = exp(input[i] - max_val);
        sum += probabilities[i];
    }

    // normalize
    for(int i = 0; i < length; ++i) {
        probabilities[i] /= sum;
    }

    return probabilities;
}

int argmax(const vector<float>& probabilities) {
    return distance(probabilities.begin(), max_element(probabilities.begin(), probabilities.end()));
}

vector<string> loadLabels(const string& labelFile) {
    vector<string> labels;
    ifstream file(labelFile);
    if (!file.is_open()) {
        cerr << "cannot open label file: " << labelFile << endl;
        return labels;
    }

    string line;
    while(getline(file, line)) {
        labels.push_back(line);
    }
    file.close();
    return labels;
}


int main(int argc, char** argv) {
    string engineFile = argv[1];
    string imageFile = argv[2];
    string labelFile = "../synset.txt"; 
    int dlaCore = 0;

    // create runtime
    IRuntime* runtime = createInferRuntime(logger);
    if (!runtime) {
        cerr << "create Runtime fail" << endl;
        return -1;
    }

    ICudaEngine* engine = loadEngine(engineFile, runtime);
    if (!engine) {
        cerr << "create engine fail" << endl;
        runtime->destroy();
        return -1;
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        cerr << "create context fail" << endl;
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    int inputH, inputW;
    float* input = preprocessImage(imageFile, inputH, inputW);
    if (!input) {
        cerr << "preprocess picture failed" << endl;
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    // input and output index
    int inputIndex = engine->getBindingIndex("data");  // model input 
    int outputIndex = engine->getBindingIndex("resnetv24_dense0_fwd"); // model output

    // cuda allocate
    void* buffers[2];
    size_t inputSize = 3 * inputH * inputW * sizeof(float);
    size_t outputSize = 1000 * sizeof(float); // 1000 class

    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);

    // create cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    float* output = new float[1000];

    // infer 10 times
    for (int i = 0; i < 10000; i++) {
      auto start = chrono::high_resolution_clock::now();
      // copy input data from host to device
      cudaMemcpyAsync(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream);
      auto end1 = chrono::high_resolution_clock::now();
      context->enqueueV2(buffers, stream, nullptr);
      cudaStreamSynchronize(stream);
      auto end2 = chrono::high_resolution_clock::now();
      // copy output data from device to host
      cudaMemcpyAsync(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      auto end3 = chrono::high_resolution_clock::now();
  
      chrono::duration<double, milli> latency1 = end3 - end2;
      chrono::duration<double, milli> latency2 = end2 - end1;
      chrono::duration<double, milli> latency3 = end1 - start;
      cout << "copy_input: " << latency1.count() << " ms" << endl;
      cout << "infer_time: " << latency2.count() << " ms" << endl;
      cout << "copy_output: " << latency3.count() << " ms" << endl;
    }

    vector<float> probabilities = softmax(output, 1000);

    // find max class
    int predictedClass = argmax(probabilities);
    float predictedProb = probabilities[predictedClass];

    // import label
    vector<string> labels = loadLabels(labelFile);
    if(labels.empty()) {
        cout << "predict class index: " << predictedClass << endl;
    } else if(predictedClass < labels.size()) {
        cout << "predict class: " << labels[predictedClass] << " (index: " << predictedClass << "), prob: " << predictedProb * 100 << "%" << endl;
    } else {
        cout << "predict class index: " << predictedClass << ", prob: " << predictedProb * 100 << "%" << endl;
    }

    cout << "Top 10 class prob:" << endl;
    vector<pair<int, float>> top10;
    for(int i = 0; i < probabilities.size(); ++i) {
        top10.emplace_back(make_pair(i, probabilities[i]));
    }
    sort(top10.begin(), top10.end(), [](const pair<int, float>& a, const pair<int, float>& b) {
        return a.second > b.second;
    });
    for(int i = 0; i < 10 && i < top10.size(); ++i) {
        int idx = top10[i].first;
        float prob = top10[i].second;
        if(labels.empty())
            cout << "class[" << idx << "] = " << prob * 100 << "%" << endl;
        else if(idx < labels.size())
            cout << "class[" << idx << "] (" << labels[idx] << ") = " << prob * 100 << "%" << endl;
        else
            cout << "class[" << idx << "] = " << prob * 100 << "%" << endl;
    }

    // release
    delete[] input;
    delete[] output;
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

#ifndef generateData_h
#define generateData_h

#include <sys/time.h>

#include <map>
#include <memory>
#include <random>

namespace Engine {
namespace CommonUtils {

template <typename T>
bool setData(T* buff, int size, T val);

template <typename T>
void read_data(T* result, size_t size, std::string filepath, char delim = ' ');

std::vector<char> read_binary(std::string filepath);
std::pair<std::shared_ptr<char>, int> read_big_binary(std::string filepath);

bool write_binary(std::string filepath, const char* buff, size_t size);

template <typename T, typename Distribution>
void GenerateRandomData(int eleSize, T* data, Distribution distribution);

bool isFileExist(const std::string filePath);
bool removeFile(const std::string filePath);
bool isDirExist(const std::string dirPath);
bool removeDir(const std::string dirPath);
bool createDir(const std::string dirPath);
std::vector<std::string> readFolder(const std::string filePath);

// read token
bool readText(std::vector<std::string>& contexts, const std::string filePath);
bool writeText(const std::vector<std::string>& contexts,
               const std::string filePath);

// paser config file with var_name=val to map<string, string>
std::map<std::string, std::string> parserKVs(
    const std::vector<std::string>& lines);

// paser config file with var_name=val to map<string, string>
std::map<std::string, std::string> parserConfig(const std::string& config_path);
std::map<std::string, std::string> parserConfig(const char* buff, size_t size);

template <typename T>
void setValueIfKeyExists(std::map<std::string, std::string>& config, T& val,
                         const std::string& key) {
  if (config.count(key)) {
    if (std::is_same<float, T>()) {
      val = std::stof(config[key]);
    } else if (std::is_same<int, T>()) {
      val = std::stoi(config[key]);
    }
  }
}

template <typename T>
bool setValueWithKey(std::map<std::string, std::string>& config, T& val,
                     const std::string& key) {
  if (config.find(key) == config.end()) return false;
  if (std::is_same<float, T>()) {
    val = std::stof(config[key]);
  } else if (std::is_same<int, T>()) {
    val = std::stoi(config[key]);
  } else if (std::is_same<bool, T>()) {
    val = std::stoi(config[key]);
  }
  return true;
}

std::vector<float> parseStringToFloatArray(const std::string& str);

// write data
template <typename T>
bool dumpFile(const T* data_ptr, size_t elementSize, std::string filePath);

static inline int64_t getTimeInUs() {
  uint64_t time;
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  return time;
}

// can be used for measuring the latency of a code snippet without adding new
// bracket
#define BENCHMARK_START(name) \
  uint64_t start_##name = CommonUtils::getTimeInUs();
#define BENCHMARK_END(name)                         \
  uint64_t end_##name = CommonUtils::getTimeInUs(); \
  EM_PRINT("[benchmark] " #name " cost: %f ms\n",   \
           float(end_##name - start_##name) / 1000.0f);

template <typename T>
float cosineSimilarity(const T* x, const T* y, int size) {
  if (x == nullptr || y == nullptr) return -1;
  double xx = 0.0, yy = 0.0;
  double cosine = 0.0;
  for (int i = 0; i < size; i++) {
    xx += x[i] * x[i];
    yy += y[i] * y[i];
    cosine += x[i] * y[i];
  }
  xx = std::sqrt(xx);
  yy = std::sqrt(yy);
  if (xx < 0.0001 || yy < 0.0001) {
    return 0;
  }
  cosine /= (xx * yy);
  return float(cosine);
}

template <typename T>
inline float interpolate1d(T x1, T y1, T x2, T y2, T x) {
  return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

std::string runSysCommand(const std::string& command);

int getMemUsage();

int fileLength(const std::string& filepath);

// pack /unpack files
bool packFiles(const std::string& dirName, const std::string& dstFile);
std::map<std::string, std::vector<char>> unPackFile(
    const std::string& packedFilePath, const std::string& dstDir);
bool unPackFile(const char* packedFileData, size_t size,
                std::map<std::string, std::vector<char>>& result);
bool unPackFile(const char* packedFileData, size_t size,
                std::map<std::string, std::pair<const char*, int>>& result);

}  // namespace CommonUtils
}  // namespace Engine

#endif /* generateData_h */
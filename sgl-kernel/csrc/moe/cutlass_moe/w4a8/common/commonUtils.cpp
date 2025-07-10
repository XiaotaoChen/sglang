#include "commonUtils.h"

#include <assert.h>
#include <dirent.h>  // dir

#include <algorithm>
#include <cstdio>   // remove
#include <cstring>  // strcmp
#include <fstream>
#include <memory>
#include <random>
#include <sstream>  // stringstream
#include <string>
#include <vector>

#include "logger.h"

template <typename T>
bool Engine::CommonUtils::setData(T* buff, int size, T val) {
  for (int i = 0; i < size; i++) {
    buff[i] = val;
  }
  return true;
}

template <typename T>
void Engine::CommonUtils::read_data(T* result, size_t size, std::string filepath,
                                    char delim) {
  std::ifstream file_handle(filepath.c_str());
  std::string line;
  int curr_count = 0;
  if (file_handle.is_open()) {
    while (getline(file_handle, line)) {
      std::stringstream ss(line);
      while (getline(ss, line, delim)) {  // split line content
        // T number = std::stof(line);
        if (std::is_same<int, T>::value) {
          result[curr_count++] = std::stoi(line);
        } else if (std::is_same<float, T>::value) {
          result[curr_count++] = std::stof(line);
        } else if (std::is_same<uint8_t, T>::value) {
          result[curr_count++] = static_cast<uint8_t>(std::stoi(line));
        } else if (std::is_same<int8_t, T>::value) {
          result[curr_count++] = static_cast<int8_t>(std::stoi(line));
        } else {
          result[curr_count++] = std::stof(line);
        }
      }
    }
    file_handle.close();
    if (size != curr_count) {
      HAI_PRINT("file:%s read size not match!\n", filepath.c_str());
    }
  } else {
    HAI_PRINT("file:%s can't open normally!\n", filepath.c_str());
  }
}

template void Engine::CommonUtils::read_data<int>(int* result, size_t size,
                                                  std::string filepath,
                                                  char delim);
template void Engine::CommonUtils::read_data<float>(float* result, size_t size,
                                                    std::string filepath,
                                                    char delim);
template void Engine::CommonUtils::read_data<uint8_t>(uint8_t* result, size_t size,
                                                      std::string filepath,
                                                      char delim);
template void Engine::CommonUtils::read_data<int8_t>(int8_t* result, size_t size,
                                                     std::string filepath,
                                                     char delim);

std::vector<float> Engine::CommonUtils::parseStringToFloatArray(
    const std::string& str) {
  std::vector<float> result;
  std::string token;
  // 移除字符串中的方括号
  std::string input = str;
  input.erase(std::remove(input.begin(), input.end(), '['), input.end());
  input.erase(std::remove(input.begin(), input.end(), ']'), input.end());
  std::stringstream ss(input);
  while (std::getline(ss, token, ',')) {
    float value = std::stof(token);
    result.push_back(value);
  }

  return result;
}

std::vector<char> Engine::CommonUtils::read_binary(std::string filepath) {
  std::ifstream input(filepath.c_str(), std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open file");
  }

  std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!input.read(buffer.data(), size)) {
    throw std::runtime_error("Failed to read file");
  }
  // std::ifstream input(filepath.c_str(), std::ios::binary);
  // std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return buffer;
}

std::pair<std::shared_ptr<char>, int> Engine::CommonUtils::read_big_binary(
    std::string filepath) {
  std::ifstream input(filepath.c_str(), std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open file");
  }

  std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);

  std::shared_ptr<char> buffer(new char[size], std::default_delete<char[]>());
  if (!input.read(buffer.get(), size)) {
    throw std::runtime_error("Failed to read file");
  }
  return {buffer, size};
}

bool Engine::CommonUtils::write_binary(std::string filepath, const char* buff,
                                       size_t size) {
  // std::fstream myfile = std::fstream(filepath.c_str(), std::ios::out |
  // std::ios::binary); myfile.write(buff, size); myfile.close();

  // write binary data with c
  FILE* fd = fopen(filepath.c_str(), "wb");
  if (fd == NULL) {
    HAI_PRINT("[write_binary] open %s failed!\n", filepath.c_str());
    return false;
  }
  fwrite(buff, sizeof(char), size, fd);
  // perror("[write_binary] fwrite error: ");
  // HAI_PRINT("[write_binary] fwrite size: %d, except size: %d, %s\n", wsize,
  // size, filepath.c_str());
  fclose(fd);
  return true;
}

// 函数具体化, 下面两种形式都可
template bool Engine::CommonUtils::setData<int>(int* buff, int size, int val);
template bool Engine::CommonUtils::setData<float>(float* buff, int size,
                                                  float val);
// template bool Engine::CommonUtils::setData(int* buff, int size, int val);
// template bool Engine::CommonUtils::setData(float* buff, int size, float val);

template <typename T, typename Distribution>
void Engine::CommonUtils::GenerateRandomData(int eleSize, T* data,
                                             Distribution distribution) {
  std::mt19937 random_engine;
  (void)std::generate_n(data, eleSize, [&]() {
    return static_cast<T>(distribution(random_engine));
  });
}

template void Engine::CommonUtils::GenerateRandomData(
    int eleSize, float* data,
    std::uniform_real_distribution<float> distribution);
template void Engine::CommonUtils::GenerateRandomData(
    int eleSize, float* data, std::normal_distribution<float> distribution);
template void Engine::CommonUtils::GenerateRandomData(
    int eleSize, int* data, std::uniform_int_distribution<int> distribution);
template void Engine::CommonUtils::GenerateRandomData(
    int eleSize, int64_t* data,
    std::uniform_int_distribution<int64_t> distribution);
template void Engine::CommonUtils::GenerateRandomData(
    int eleSize, uint8_t* data,
    std::uniform_int_distribution<uint8_t> distribution);
template void Engine::CommonUtils::GenerateRandomData(
    int eleSize, int8_t* data,
    std::uniform_int_distribution<int8_t> distribution);

// file operator
bool Engine::CommonUtils::isFileExist(const std::string filePath) {
  if (FILE* file = fopen(filePath.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

std::vector<std::string> Engine::CommonUtils::readFolder(
    const std::string image_path) {
  std::vector<std::string> res;
  auto dir = opendir(image_path.c_str());
  if (dir != NULL) {
    struct dirent* entry;
    entry = readdir(dir);
    while (entry) {
      auto temp = entry->d_name;
      if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 ||
          strcmp(entry->d_name, "..") == 0) {
        entry = readdir(dir);
        continue;
      }
      res.push_back(temp);
      entry = readdir(dir);
    }
  }
  closedir(dir);
  return res;
}

bool Engine::CommonUtils::removeFile(const std::string filePath) {
  if (remove(filePath.c_str()) == 0) {
    return true;
  }
  return false;
}

bool Engine::CommonUtils::isDirExist(const std::string dirPath) {
  DIR* root = opendir(dirPath.c_str());
  if (NULL == root) {
    return false;
  }
  closedir(root);
  return true;
}

bool Engine::CommonUtils::removeDir(const std::string dirPath) {
  std::string cmd = "rm -rf " + dirPath;
  system(cmd.c_str());
  return true;
}

bool Engine::CommonUtils::createDir(const std::string dirPath) {
  std::string cmd = "mkdir -p " + dirPath;
  system(cmd.c_str());
  return true;
}

bool Engine::CommonUtils::readText(std::vector<std::string>& contexts,
                                   const std::string filePath) {
  std::ifstream file_handle(filePath.c_str());
  std::string line;
  if (file_handle.is_open()) {
    while (getline(file_handle, line)) {
      contexts.emplace_back(line);
    }
    file_handle.close();
  } else {
    HAI_PRINT("file %s can't open normally!\n", filePath.c_str());
    return false;
  }
  return true;
}

bool Engine::CommonUtils::writeText(const std::vector<std::string>& contexts,
                                    const std::string filePath) {
  std::ofstream file_handle(filePath.c_str());
  if (file_handle.is_open()) {
    for (const auto& item : contexts) {
      file_handle << item + "\n";
    }
    file_handle.close();
    return true;
  }
  return false;
}

template <typename T>
bool Engine::CommonUtils::dumpFile(const T* data_ptr, size_t elementSize,
                                   std::string filePath) {
  std::ofstream outputOs(filePath.c_str());
  if (outputOs.is_open()) {
    if (std::is_same<int, T>::value || std::is_same<float, T>::value) {
      for (int i = 0; i < elementSize; i++) {
        outputOs << data_ptr[i] << "\n";
      }
    } else if (std::is_same<uint8_t, T>::value ||
               std::is_same<int8_t, T>::value) {
      for (int i = 0; i < elementSize; i++) {
        outputOs << static_cast<int>(data_ptr[i]) << "\n";
      }
    }
    outputOs.close();
    return true;
  }
  return false;
}

template bool Engine::CommonUtils::dumpFile(const float* data_ptr,
                                            size_t elementSize,
                                            std::string filePath);

template bool Engine::CommonUtils::dumpFile(const int* data_ptr,
                                            size_t elementSize,
                                            std::string filePath);
template bool Engine::CommonUtils::dumpFile(const uint8_t* data_ptr,
                                            size_t elementSize,
                                            std::string filePath);
template bool Engine::CommonUtils::dumpFile(const int8_t* data_ptr,
                                            size_t elementSize,
                                            std::string filePath);

std::map<std::string, std::string> Engine::CommonUtils::parserKVs(
    const std::vector<std::string>& lines) {
  std::map<std::string, std::string> result;
  std::string key;
  std::string val;
  for (size_t i = 0; i < lines.size(); i++) {
    std::size_t idx = lines[i].find_first_of('=');
    if (idx != std::string::npos) {
      key = lines[i].substr(0, idx);
      val = lines[i].substr(idx + 1);
      result[key] = val;
    }
  }
  return result;
}

std::map<std::string, std::string> Engine::CommonUtils::parserConfig(
    const std::string& config_path) {
  std::vector<std::string> lines;
  bool flag = readText(lines, config_path);
  if (flag == false) return {};
  return parserKVs(lines);
}

std::map<std::string, std::string> Engine::CommonUtils::parserConfig(
    const char* buff, size_t size) {
  std::string cfg_contents(buff, size);
  std::vector<std::string> lines;
  while (cfg_contents.length() > 0) {
    size_t idx = cfg_contents.find_first_of('\n');
    if (idx != std::string::npos) {
      lines.push_back(cfg_contents.substr(0, idx));
      if (idx < cfg_contents.length()) {
        cfg_contents = cfg_contents.substr(idx + 1);
      } else {
        cfg_contents.clear();
      }
    } else {
      lines.push_back(cfg_contents);
      cfg_contents.clear();
    }
  }
  return parserKVs(lines);
}

std::string Engine::CommonUtils::runSysCommand(const std::string& command) {
  FILE* fp;
  char buffer[100];
  fp = popen(command.c_str(), "r");
  fgets(buffer, sizeof(buffer), fp);
  pclose(fp);
  return std::string(buffer);
}

int Engine::CommonUtils::getMemUsage() {
#ifndef USE_AX
  return -1;
#endif

  std::string memInfoPath = "/proc/ax_proc/mem_cmm_info";
  std::ifstream ifs(memInfoPath);
  if (!ifs.is_open()) {
    return -1;
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.find("used=") != std::string::npos) {
      std::istringstream iss(line);
      std::string key;
      std::string mems;
      iss >> key >> mems;
      auto pos = mems.find("used=");
      auto endPos = mems.find("KB", pos);
      pos += 5;  // length of "used="
      int value = std::stoi(mems.substr(pos, endPos - pos));

      return value;
    }
  }

  return -1;
}

int Engine::CommonUtils::fileLength(const std::string& filepath) {
  int length;
  std::ifstream is;
  is.open(filepath.c_str(), std::ios::binary);
  // get length of file:
  is.seekg(0, std::ios::end);
  length = is.tellg();
  is.close();
  return length;
}

bool Engine::CommonUtils::packFiles(const std::string& dirName,
                                    const std::string& dstFile) {
  HAI_PRINT("******* to pack directory: %s *******\n", dirName.c_str());
  std::vector<std::string> files = Engine::CommonUtils::readFolder(dirName);
  for (const auto& file : files) {
    if (Engine::CommonUtils::isDirExist(dirName + "/" + file)) {
      HAI_PRINT("%s include %s is directory: can't to pack it\n",
               dirName.c_str(), file.c_str());
      return false;
    }
  }

  HAI_PRINT("total file count: %zu\n", files.size());

  // 100MB
  int baseSize = 100 * 1024 * 1024;
  // pack format: file_numbers, [filename_length, file_length],
  // [filename_contents, file_contents]

  // save file_numbers, filename_length, fil_length
  int baseContentSize = (1 + files.size() * 2) * sizeof(int);
  std::unique_ptr<char[]> baseContent(new char[baseContentSize]);
  char* curr_ptr = baseContent.get();
  int size = files.size();
  std::memcpy(curr_ptr, &size, sizeof(int));
  curr_ptr += sizeof(int);

  int contentSize = baseContentSize;
  // filename length
  for (size_t i = 0; i < files.size(); i++) {
    int size = files[i].length();
    std::memcpy(curr_ptr, &size, sizeof(int));
    curr_ptr += sizeof(int);
    size = Engine::CommonUtils::fileLength(dirName + "/" + files[i]);
    std::memcpy(curr_ptr, &size, sizeof(int));
    curr_ptr += sizeof(int);

    contentSize += files[i].length();
    contentSize += size;

    HAI_PRINT("%s len: %zu content size: %d\n", files[i].c_str(),
             files[i].length(), size);
  }

  HAI_PRINT("total contentSize size: %d\n", contentSize);

  // file contents length
  int targetSize = baseSize;
  while (contentSize > targetSize) {
    targetSize += baseSize;
  }

  std::unique_ptr<char[]> buffer(new char[contentSize]);
  std::memcpy(buffer.get(), baseContent.get(), baseContentSize);
  curr_ptr = buffer.get() + baseContentSize;

  int fileSize = 0;
  for (const auto& file : files) {
    std::string filePath = dirName + "/" + file;
    if (Engine::CommonUtils::isFileExist(filePath)) {
      // filename content
      std::memcpy(curr_ptr, file.c_str(), file.length());
      curr_ptr += file.length();
      fileSize += file.length();
      // file content
      std::vector<char> contents = Engine::CommonUtils::read_binary(filePath);
      std::memcpy(curr_ptr, contents.data(), contents.size());
      curr_ptr += contents.size();
      fileSize += contents.size();
    } else {
      HAI_PRINT("%s is not exist\n", file.c_str());
      return false;
    }
  }

  if (contentSize != fileSize + baseContentSize) {
    HAI_PRINT("pre calculate packed content size: %d, mismatch with %d\n",
             contentSize, fileSize + baseContentSize);
  }

  bool isOk =
      Engine::CommonUtils::write_binary(dstFile, buffer.get(), contentSize);

  if (!isOk) {
    HAI_PRINT("write binary failed\n");
    return false;
  }
  return true;
}

bool Engine::CommonUtils::unPackFile(
    const char* packedFileData, size_t /*size*/,
    std::map<std::string, std::vector<char>>& result) {
  const int* int_ptr = (const int*)packedFileData;
  int fileCount = *int_ptr;
  int_ptr++;
  std::vector<std::pair<int, int>> fileNameContentCounts(fileCount);
  // file name , file content size
  for (int i = 0; i < fileCount; i++) {
    fileNameContentCounts[i].first = *int_ptr;
    int_ptr++;
    fileNameContentCounts[i].second = *int_ptr;
    int_ptr++;
  }

  const char* curr_ptr = (const char*)int_ptr;
  // file name, content
  for (int i = 0; i < fileCount; i++) {
    // file name
    std::string key = std::string(curr_ptr, fileNameContentCounts[i].first);
    curr_ptr += fileNameContentCounts[i].first;
    // file content
    result[key] = std::vector<char>(fileNameContentCounts[i].second);
    std::memcpy(result[key].data(), curr_ptr, fileNameContentCounts[i].second);
    curr_ptr += fileNameContentCounts[i].second;
  }

  return true;
}

bool Engine::CommonUtils::unPackFile(
    const char* packedFileData, size_t /*size*/,
    std::map<std::string, std::pair<const char*, int>>& result) {
  const int* int_ptr = (const int*)packedFileData;
  int fileCount = *int_ptr;
  int_ptr++;
  std::vector<std::pair<int, int>> fileNameContentCounts(fileCount);
  // file name , file content size
  for (int i = 0; i < fileCount; i++) {
    fileNameContentCounts[i].first = *int_ptr;
    int_ptr++;
    fileNameContentCounts[i].second = *int_ptr;
    int_ptr++;
  }

  const char* curr_ptr = (const char*)int_ptr;
  // file name, content
  for (int i = 0; i < fileCount; i++) {
    // file name
    std::string key = std::string(curr_ptr, fileNameContentCounts[i].first);
    curr_ptr += fileNameContentCounts[i].first;
    // file content
    result[key] = {curr_ptr, fileNameContentCounts[i].second};
    curr_ptr += fileNameContentCounts[i].second;
  }
  return true;
}

std::map<std::string, std::vector<char>> Engine::CommonUtils::unPackFile(
    const std::string& packedFilePath, const std::string& dstDir) {
  if (!Engine::CommonUtils::isFileExist(packedFilePath)) {
    HAI_PRINT("packed file: %s not exist\n", packedFilePath.c_str());
    return {};
  }
  std::vector<char> buff = Engine::CommonUtils::read_binary(packedFilePath);
  char* curr_ptr = buff.data();

  std::map<std::string, std::vector<char>> fileContents;
  bool isOk =
      Engine::CommonUtils::unPackFile(curr_ptr, buff.size(), fileContents);
  if (isOk == false) {
    HAI_PRINT("[Engine::CommonUtils::unPackFile] unpack file failed\n");
    return {};
  }

  // save contents
  if (dstDir.empty() == false && dstDir != "") {
    if (!Engine::CommonUtils::isDirExist(dstDir)) {
      Engine::CommonUtils::createDir(dstDir);
    }
    for (const auto& pair : fileContents) {
      std::string dstPath = dstDir + "/" + pair.first;
      if (!Engine::CommonUtils::write_binary(dstPath, pair.second.data(),
                                             pair.second.size())) {
        HAI_PRINT("dump %s failed\n", dstPath.c_str());
        return {};
      }
    }
  }

  return fileContents;
}

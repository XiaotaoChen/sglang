//
//  logger.h
//  logger
//
//

#ifndef logger_h
#define logger_h

#include <stdio.h>

#define HAI_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define HAI_ERROR(format, ...) printf(format, ##__VA_ARGS__)

#define HAI_FUNC_PRINT(x) HAI_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define HAI_FUNC_PRINT_ALL(x, type) \
  HAI_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define HAI_ASSERT(x)                                      \
  {                                                       \
    int res = (x);                                        \
    if (!res) {                                           \
      HAI_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
    }                                                     \
  }
#endif /* logger_h */
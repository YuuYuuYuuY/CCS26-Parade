#pragma once

#include <string>

extern "C" {
#include <bgpstream_utils_as_path.h>
}

inline std::string exact_as_path_key(bgpstream_as_path_t* path) {
    uint8_t* bytes = nullptr;
    const auto length = bgpstream_as_path_get_data(path, &bytes);
    if (!length) return {};
    return std::string(reinterpret_cast<const char*>(bytes), length);
}

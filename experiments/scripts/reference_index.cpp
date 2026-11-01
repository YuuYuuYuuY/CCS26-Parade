#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
    std::vector<char> buffer(8 * 1024 * 1024);
    std::ifstream input;
    input.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
    input.open(argv[1], std::ios::binary);
    uint64_t total = 0;
    input.read(reinterpret_cast<char*>(&total), sizeof(total));
    std::vector<std::vector<uint64_t>> offsets;
    std::unordered_set<uint32_t> ases;
    std::unordered_set<uint64_t> edges;
    ases.reserve(100000);
    edges.reserve(700000);
    std::array<uint32_t, 255> path{};
    std::array<unsigned char, 5> header{};
    uint64_t offset = 8, prefix_count = 0;
    for (uint64_t record = 0; record < total; ++record) {
        input.read(reinterpret_cast<char*>(header.data()), header.size());
        int32_t prefix;
        std::memcpy(&prefix, header.data(), 4);
        const uint8_t length = header[4];
        input.read(reinterpret_cast<char*>(path.data()), length*sizeof(uint32_t));
        if (size_t(prefix) >= offsets.size()) offsets.resize(size_t(prefix)+1);
        auto& bucket = offsets[prefix];
        if (bucket.empty()) ++prefix_count;
        bucket.push_back(offset);
        offset += 5 + uint64_t(length)*4;
        for (unsigned position = 0; position < length; ++position) {
            const uint32_t asn = path[position];
            ases.insert(asn);
            if (position && asn != path[position-1]) {
                const auto left = std::min(asn, path[position-1]);
                const auto right = std::max(asn, path[position-1]);
                edges.insert((uint64_t(left) << 32) | right);
            }
        }
    }
    input.close();
    std::vector<uint32_t> ordered_ases(ases.begin(), ases.end());
    std::vector<uint64_t> ordered_edges(edges.begin(), edges.end());
    std::sort(ordered_ases.begin(), ordered_ases.end());
    std::sort(ordered_edges.begin(), ordered_edges.end());
    const std::string source = std::filesystem::absolute(argv[1]).string();
    const uint32_t source_size = source.size();
    const uint64_t summary[] = {offset, total, prefix_count, ordered_ases.size(), ordered_edges.size()};
    std::ofstream out(argv[2], std::ios::binary);
    out.write("PREFIDX2", 8);
    out.write(reinterpret_cast<const char*>(summary), sizeof(summary));
    out.write(reinterpret_cast<const char*>(&source_size), sizeof(source_size));
    out.write(source.data(), source.size());
    uint64_t flattened_start = 0;
    for (size_t prefix = 0; prefix < offsets.size(); ++prefix) {
        const auto& bucket = offsets[prefix];
        if (bucket.empty()) continue;
        const int32_t prefix_id = prefix;
        const uint64_t range[] = {flattened_start, bucket.size()};
        out.write(reinterpret_cast<const char*>(&prefix_id), sizeof(prefix_id));
        out.write(reinterpret_cast<const char*>(range), sizeof(range));
        flattened_start += bucket.size();
    }
    for (const auto& bucket : offsets)
        out.write(reinterpret_cast<const char*>(bucket.data()), bucket.size()*sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(ordered_ases.data()), ordered_ases.size()*sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(ordered_edges.data()), ordered_edges.size()*sizeof(uint64_t));
}

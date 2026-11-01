#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
struct Header { uint32_t profiles, ases; uint64_t rows; };
struct Entry { uint32_t profile, hop; };

struct Mapping {
    int fd;
    size_t bytes;
    void* memory;
    Mapping(const char* path, bool write, size_t size = 0) {
        fd = open(path, write ? O_RDWR | O_CREAT | O_TRUNC : O_RDONLY, 0644);
        if (write) ftruncate(fd, size);
        struct stat st{};
        fstat(fd, &st);
        bytes = st.st_size;
        memory = mmap(nullptr, bytes, write ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
    }
    ~Mapping() { munmap(memory, bytes); close(fd); }
};

struct Index : Mapping {
    const Header* header;
    const uint64_t* offsets;
    const uint64_t* fingerprints;
    const Entry* entries;
    Index(const char* path) : Mapping(path, false) {
        header = static_cast<const Header*>(memory);
        offsets = reinterpret_cast<const uint64_t*>(header + 1);
        fingerprints = offsets + header->ases + 1;
        entries = reinterpret_cast<const Entry*>(fingerprints + uint64_t(header->profiles) * 2);
    }
};
}

extern "C" {
void ch_build(const double* rows, uint64_t count, const uint64_t* fingerprints,
              uint32_t profiles, uint32_t ases, const char* path) {
    std::vector<uint64_t> offsets(uint64_t(ases) + 1, 0);
    for (uint64_t i = 0; i < count; ++i) ++offsets[uint32_t(rows[i * 4 + 1]) + 1];
    for (uint32_t aid = 0; aid < ases; ++aid) offsets[aid + 1] += offsets[aid];
    Mapping output(path, true, sizeof(Header) + offsets.size() * 8 + uint64_t(profiles) * 16 + count * sizeof(Entry));
    auto* header = static_cast<Header*>(output.memory);
    *header = {profiles, ases, count};
    auto* target_offsets = reinterpret_cast<uint64_t*>(header + 1);
    memcpy(target_offsets, offsets.data(), offsets.size() * 8);
    auto* target_fp = target_offsets + offsets.size();
    memcpy(target_fp, fingerprints, uint64_t(profiles) * 16);
    auto* entries = reinterpret_cast<Entry*>(target_fp + uint64_t(profiles) * 2);
    for (uint64_t i = 0; i < count; ++i) {
        const double* row = rows + i * 4;
        entries[offsets[uint32_t(row[1])]++] = {uint32_t(row[0]), uint32_t(row[2])};
    }
}

void* ch_open(const char* path) { return new Index(path); }
void ch_close(void* handle) { delete static_cast<Index*>(handle); }

int ch_eligible(void* handle, uint32_t pid, uint32_t aid) {
    const auto& index = *static_cast<Index*>(handle);
    const Entry* begin = index.entries + index.offsets[aid];
    const Entry* end = index.entries + index.offsets[aid + 1];
    const Entry* direct = std::lower_bound(begin, end, pid,
        [](const Entry& entry, uint32_t value) { return entry.profile < value; });
    if (direct != end && direct->profile == pid && direct->hop < 8) return 0;
    const uint64_t* query = index.fingerprints + uint64_t(pid) * 2;
    for (auto entry = begin; entry != end; ++entry) {
        if (entry->profile == pid || entry->hop >= 4) continue;
        const uint64_t* other = index.fingerprints + uint64_t(entry->profile) * 2;
        unsigned distance = __builtin_popcountll(query[0] ^ other[0]) + __builtin_popcountll(query[1] ^ other[1]);
        if (distance < 32) return 0;
    }
    return 1;
}
}

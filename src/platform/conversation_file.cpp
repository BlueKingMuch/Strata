// The streaming envelope/integrity-footer design follows @maedoc's Strata #52.
// Serialize the shared image instead of its independent CUDA segment/apply walk.
#include "strata/platform/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <share.h>
#else
#include <sys/types.h>
#endif

namespace strata::platform {
namespace {
using core::ConversationCheckpoint;
using core::ConversationImageKey;
using core::ConversationKv;
using core::ConversationBuffer;
using core::SavedConversation;
constexpr std::array<uint8_t, 8> magic{'S','T','R','S','N','A','P',2};
// GPU state blobs retain their native scalar representation.
static_assert(std::endian::native == std::endian::little);
// These portable allowances also bound vector-object storage before resize.
constexpr uint64_t checkpoint_overhead = 512, kv_overhead = 256;
static_assert(sizeof(ConversationCheckpoint) <= checkpoint_overhead);
static_assert(sizeof(ConversationKv) <= kv_overhead);
static_assert(sizeof(SavedConversation) <= 1024);

struct Digest {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(), EVP_MD_CTX_free};
    Digest() {
        if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("SHA-256 initialization failed");
    }
    void update(const void* p, size_t n) {
        if (n && EVP_DigestUpdate(ctx.get(), p, n) != 1)
            throw std::runtime_error("SHA-256 update failed");
    }
    ConversationDigest finish() {
        ConversationDigest out;
        unsigned n = 0;
        if (EVP_DigestFinal_ex(ctx.get(), out.data(), &n) != 1 || n != out.size())
            throw std::runtime_error("SHA-256 finalization failed");
        return out;
    }
};

std::array<uint8_t, 8> little(uint64_t value) {
    std::array<uint8_t, 8> bytes;
    for (unsigned i = 0; i < 8; ++i) bytes[i] = uint8_t(value >> (8 * i));
    return bytes;
}
void add(uint64_t& total, uint64_t count, uint64_t width = 1) {
    if (width && count > (UINT64_MAX - total) / width)
        throw std::runtime_error("snapshot allocation count overflow");
    total += count * width;
}
uint64_t allocation_bound(const SavedConversation& image) {
    uint64_t total = kConversationFileWorkspace;
    add(total, image.checkpoints.size(), checkpoint_overhead);
    add(total, image.kv.size(), kv_overhead);
    auto checkpoint = [&](const ConversationCheckpoint& c) {
        if (!c.stage_parts.empty()) throw std::runtime_error("layer-split snapshots are unsupported");
        add(total, c.ids.size(), sizeof(int32_t));
        add(total, c.imgs.size(), sizeof(ConversationImageKey));
        for (const auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) add(total, v->size());
    };
    checkpoint(image.live);
    for (const auto& c : image.checkpoints) checkpoint(c);
    for (const auto& kv : image.kv)
        for (const auto* v : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) {
            add(total, v->size());
            // The reader rebuilds canonical segments, independently of how many
            // small appends produced the in-memory image. Allow portable directory storage.
            add(total, v->size() / ConversationBuffer::segment_bytes +
                       (v->size() % ConversationBuffer::segment_bytes != 0), 64);
        }
    return total;
}

struct IoProgress {
    ConversationIoProgress callback = nullptr;
    uint64_t bytes = 0;
    void advance(uint64_t n) {
        if (!callback) return;
        if (n >= (1 << 20) - bytes) { bytes = 0; callback(); }
        else bytes += n;
    }
    void finish() { if (callback) callback(); }
};

// A file read at given offsets through C stdio, one fread per piece. Not std::ifstream: MSVC's filebuf::xsgetn hands
// fread 4,095 bytes at a time, so every read became 4 KiB ReadFile calls, about 0.8 GB/s per stream.
class InputFile {
public:
    explicit InputFile(const std::filesystem::path& path) {
#ifdef _WIN32
        file_ = _wfsopen(path.c_str(), L"rb", _SH_DENYNO);   // the sharing std::ifstream uses
#else
        file_ = std::fopen(path.c_str(), "rb");
#endif
    }
    ~InputFile() { if (file_) std::fclose(file_); }
    InputFile(const InputFile&) = delete;
    InputFile& operator=(const InputFile&) = delete;
    bool is_open() const { return file_ != nullptr; }
    bool read(uint64_t offset, void* data, size_t count) {
        if (!file_ || offset > uint64_t(std::numeric_limits<int64_t>::max())) return false;
#ifdef _WIN32
        if (_fseeki64(file_, (long long) offset, SEEK_SET) != 0) return false;
#else
        if (fseeko(file_, (off_t) offset, SEEK_SET) != 0) return false;
#endif
        return std::fread(data, 1, count, file_) == count;
    }
private:
    std::FILE* file_ = nullptr;
};

// Large spans (state blobs, K/V segments) move in 4 MiB pieces, and one piece is hashed while the stream writes
// or reads its neighbour, so hashing no longer adds to the time the stream needs. The bytes and the digest are the
// same as before; only small fields still go through one piece at a time.
constexpr size_t kPipelinePiece = size_t(4) << 20;

struct Writer {
    std::ostream& stream;
    Digest digest;
    IoProgress progress;
    void bytes(const void* p, size_t n) {
        const auto* data = static_cast<const char*>(p);
        if (n >= 2 * kPipelinePiece) {
            std::future<bool> written;   // the previous piece; the stream is only ever used by one thread at a time
            for (size_t done = 0; done < n;) {
                const size_t count = std::min(n - done, kPipelinePiece);
                digest.update(data + done, count);
                progress.advance(count);
                if (written.valid() && !written.get()) throw std::runtime_error("snapshot write failed");
                written = std::async(std::launch::async, [this, at = data + done, count] {
                    return bool(stream.write(at, std::streamsize(count)));
                });
                done += count;
            }
            if (!written.get()) throw std::runtime_error("snapshot write failed");
            return;
        }
        // Keep streamsize conversion bounded even for a very large snapshot.
        while (n) {
            const size_t chunk = std::min<size_t>(n, 65536);
            if (!stream.write(data, chunk)) throw std::runtime_error("snapshot write failed");
            digest.update(data, chunk);
            progress.advance(chunk);
            data += chunk; n -= chunk;
        }
    }
    void integer(uint64_t value) { const auto b = little(value); bytes(b.data(), b.size()); }
    void blob(const std::vector<uint8_t>& v) { integer(v.size()); bytes(v.data(), v.size()); }
    void blob(const ConversationBuffer& v) {
        integer(v.size());
        v.visit(0, v.size(), [&](const uint8_t* p, size_t n, size_t) { bytes(p,n); return true; });
    }
    void checkpoint(const ConversationCheckpoint& c) {
        integer(c.ids.size());
        for (int32_t id : c.ids) {
            const auto b = little(std::bit_cast<uint32_t>(id)); bytes(b.data(), 4);
        }
        integer(c.imgs.size());
        for (const auto& i : c.imgs) { integer(std::bit_cast<uint64_t>(i.start)); integer(i.hash); }
        integer(c.used);
        for (const auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) blob(*v);
    }
};

// Reading a snapshot from a file, its spans of 1 MiB and more are read by reader threads: each 1 MiB piece with one
// fread from that thread's own handle straight into the image, and hashed in file order on the calling thread as the
// pieces arrive. One stream read about 0.8 GB/s on Windows; several reach the drive's speed, and the hash (about
// 2 GB/s with SHA extensions) becomes the limit. The parser still sees the bytes in order, the digest is the same, no
// second buffer exists, and no read outlives the span it belongs to.
constexpr size_t kSpanPiece = size_t(1) << 20;

class SpanReader {
public:
    using Fragment = std::pair<char*, size_t>;
    SpanReader(std::filesystem::path path, unsigned threads) : path_(std::move(path)) {
        try {
            for (unsigned t = 0; t < threads; ++t) threads_.emplace_back([this] { work(); });
        } catch (const std::system_error&) {}   // fewer threads; with none, the caller reads the stream itself
    }
    ~SpanReader() {
        { std::lock_guard<std::mutex> lock(mu_); stop_ = true; }
        work_cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    SpanReader(const SpanReader&) = delete;
    SpanReader& operator=(const SpanReader&) = delete;
    size_t threads() const { return threads_.size(); }

    // Reads the file's bytes from `offset` on into the fragments, which are consecutive in the file, and hands each
    // piece to in_order on this thread in file order. Every way out waits until no thread is still reading a piece:
    // the fragments belong to the caller.
    void read(uint64_t offset, const std::vector<Fragment>& fragments, const std::function<void(const char*, size_t)>& in_order) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            pieces_.clear();
            for (const auto& [data, n] : fragments)
                for (size_t done = 0; done < n; done += kSpanPiece) {
                    const size_t count = std::min(kSpanPiece, n - done);
                    pieces_.push_back({offset, data + done, count});
                    offset += count;
                }
            state_.assign(pieces_.size(), 0);
            next_ = 0;
            abort_ = false;
        }
        work_cv_.notify_all();
        struct Settle {
            SpanReader& r;
            ~Settle() {
                std::unique_lock<std::mutex> lock(r.mu_);
                r.abort_ = true;   // no further claims
                r.done_cv_.wait(lock, [&] { return r.busy_ == 0; });
                r.pieces_.clear(); r.state_.clear(); r.next_ = 0;
            }
        } settle{*this};
        for (size_t i = 0; i < pieces_.size(); ++i) {   // pieces_ only changes before and after the span
            {
                std::unique_lock<std::mutex> lock(mu_);
                done_cv_.wait(lock, [&] { return state_[i] != 0; });
                if (state_[i] != 1) throw std::runtime_error("truncated snapshot");
            }
            in_order(pieces_[i].data, pieces_[i].count);
        }
    }

private:
    struct Piece { uint64_t offset; char* data; size_t count; };
    void work() {
        std::unique_ptr<InputFile> file;   // opened on the first piece, kept for the read
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            work_cv_.wait(lock, [&] { return stop_ || (!abort_ && next_ < pieces_.size()); });
            if (stop_) return;
            const size_t i = next_++;
            const Piece piece = pieces_[i];
            ++busy_;
            lock.unlock();
            if (!file) file = std::make_unique<InputFile>(path_);
            const bool ok = file->read(piece.offset, piece.data, piece.count);
            lock.lock();
            state_[i] = uint8_t(ok ? 1 : 2);
            --busy_;
            done_cv_.notify_all();
        }
    }
    const std::filesystem::path path_;
    std::mutex mu_;
    std::condition_variable work_cv_, done_cv_;
    std::vector<Piece> pieces_;
    std::vector<uint8_t> state_;   // per piece: 0 not yet read, 1 read, 2 failed
    size_t next_ = 0, busy_ = 0;   // the next piece to claim; pieces being read
    bool stop_ = false, abort_ = false;
    std::vector<std::thread> threads_;
};

struct Reader {
    std::istream& stream;
    Digest digest;
    uint64_t remaining = 0;
    IoProgress progress;
    const std::filesystem::path* source = nullptr;   // the file behind the stream, for spans read on several threads
    unsigned threads = 1;
    uint64_t offset = 0;                              // where the stream is: the bytes consumed so far
    std::unique_ptr<SpanReader> spans{};

    bool parallel(size_t n) {
        if (!source || n < kSpanPiece) return false;
        if (!spans) spans = std::make_unique<SpanReader>(*source, std::max(1u, threads));
        return spans->threads() > 0;
    }
    // A span read on several threads; the stream then continues after it.
    void span(const std::vector<SpanReader::Fragment>& fragments, size_t n) {
        spans->read(offset, fragments, [this](const char* p, size_t count) {
            digest.update(p, count);
            progress.advance(count);
        });
        offset += n;
        if (!stream.seekg(std::streamoff(offset))) throw std::runtime_error("truncated snapshot");
    }
    void bytes(void* p, size_t n) {
        auto* data = static_cast<char*>(p);
        if (parallel(n)) { span({{data, n}}, n); return; }
        offset += n;
        if (n >= 2 * kPipelinePiece) {
            const auto fill = [this](char* at, size_t count) { return bool(stream.read(at, std::streamsize(count))); };
            size_t done = 0, count = kPipelinePiece;
            if (!fill(data, count)) throw std::runtime_error("truncated snapshot");
            while (true) {
                const size_t next_at = done + count, next = std::min(n - next_at, kPipelinePiece);
                std::future<bool> pending;   // the stream is only ever used by one thread at a time
                if (next) pending = std::async(std::launch::async, fill, data + next_at, next);
                digest.update(data + done, count);
                progress.advance(count);
                if (!next) break;
                if (!pending.get()) throw std::runtime_error("truncated snapshot");
                done = next_at; count = next;
            }
            return;
        }
        while (n) {
            const size_t chunk = std::min<size_t>(n, 65536);
            if (!stream.read(data, chunk)) throw std::runtime_error("truncated snapshot");
            digest.update(data, chunk);
            progress.advance(chunk);
            data += chunk; n -= chunk;
        }
    }
    uint64_t integer(size_t width = 8) {
        std::array<uint8_t, 8> b{}; bytes(b.data(), width);
        uint64_t out = 0;
        for (size_t i = 0; i < width; ++i) out |= uint64_t(b[i]) << (8 * i);
        return out;
    }
    int64_t signed_integer() { return std::bit_cast<int64_t>(integer()); }
    template<class T> void allocate(std::vector<T>& v, uint64_t n, uint64_t width = sizeof(T)) {
        if (n > v.max_size() || n > remaining / width)
            throw std::runtime_error("snapshot exceeds admitted staging bytes");
        remaining -= n * width;
        v.resize(static_cast<size_t>(n));
        // New vectors normally allocate exactly n elements; account for any excess.
        const uint64_t extra = (v.capacity() - v.size()) * sizeof(T);
        if (extra > remaining) throw std::runtime_error("snapshot vector capacity exceeds staging bound");
        remaining -= extra;
    }
    void blob(std::vector<uint8_t>& v) { allocate(v, integer()); bytes(v.data(), v.size()); }
    void blob(ConversationBuffer& v) {
        const uint64_t n = integer();
        if (n > SIZE_MAX) throw std::runtime_error("snapshot buffer size overflow");
        const size_t admitted = v.allocation_peak(size_t(n));
        if (admitted == SIZE_MAX || admitted > remaining)
            throw std::runtime_error("snapshot segmented buffer exceeds staging bound");
        remaining -= admitted;
        v.resize(size_t(n));
        if (v.bytes() > admitted) {
            const size_t extra = v.bytes() - admitted;
            if (extra > remaining) throw std::runtime_error("snapshot segment capacity exceeds staging bound");
            remaining -= extra;
        }
        if (parallel(v.size())) {   // all its segments as one span, so the threads read ahead across them
            std::vector<SpanReader::Fragment> fragments;
            v.visit(0, v.size(), [&](uint8_t* p, size_t count, size_t) {
                fragments.emplace_back(reinterpret_cast<char*>(p), count);
                return true;
            });
            span(fragments, v.size());
            return;
        }
        v.visit(0, v.size(), [&](uint8_t* p, size_t count, size_t) { bytes(p,count); return true; });
    }
    void checkpoint(ConversationCheckpoint& c) {
        allocate(c.ids, integer());
        for (auto& id : c.ids) id = std::bit_cast<int32_t>(static_cast<uint32_t>(integer(4)));
        allocate(c.imgs, integer());
        for (auto& i : c.imgs) { i.start = signed_integer(); i.hash = integer(); }
        c.used = integer();
        for (auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) blob(*v);
    }
};

// Candidate selection reads only prefix metadata. Full decoding checks integrity of
// the selected file before the shared core can apply any state.
struct Probe {
    std::istream& stream;
    uint64_t file_remaining, allocation_remaining;
    IoProgress progress;
    void consume(uint64_t n) {
        if (n > file_remaining) throw std::runtime_error("truncated snapshot metadata");
        file_remaining -= n;
    }
    void bytes(void* p, size_t n) {
        consume(n);
        if (!stream.read(static_cast<char*>(p), n)) throw std::runtime_error("snapshot metadata read failed");
        progress.advance(n);
    }
    void skip(uint64_t n) {
        consume(n);
        if (n > uint64_t(std::numeric_limits<std::streamoff>::max()) ||
            !stream.seekg(static_cast<std::streamoff>(n), std::ios::cur))
            throw std::runtime_error("snapshot metadata seek failed");
        progress.advance(n);
    }
    uint64_t integer(size_t width = 8) {
        std::array<uint8_t, 8> b{}; bytes(b.data(), width);
        uint64_t out = 0;
        for (size_t i = 0; i < width; ++i) out |= uint64_t(b[i]) << (8 * i);
        return out;
    }
    void account(uint64_t n, uint64_t width) {
        if (n > allocation_remaining / width)
            throw std::runtime_error("snapshot metadata exceeds staging bound");
        allocation_remaining -= n * width;
    }
    int64_t checkpoint(const std::vector<int64_t>& prompt, const std::vector<ConversationImageKey>& images) {
        const uint64_t count = integer(); account(count, sizeof(int32_t));
        bool equal = count != 0 && count < prompt.size();
        if (equal) {
            for (size_t i = 0; i < count; ++i)
                if (std::bit_cast<int32_t>(static_cast<uint32_t>(integer(4))) != prompt[i]) equal = false;
        } else skip(count * 4);
        const uint64_t image_count = integer(); account(image_count, sizeof(ConversationImageKey));
        if (equal) {
            const size_t expected = std::count_if(images.begin(), images.end(),
                [&](const auto& image) { return image.start < static_cast<int64_t>(count); });
            equal = image_count == expected;
        }
        if (equal) {
            for (const auto& image : images) {
                if (image.start >= static_cast<int64_t>(count)) continue;
                const int64_t start = std::bit_cast<int64_t>(integer());
                const uint64_t hash = integer();
                if (image.start != start || image.hash != hash) equal = false;
            }
        } else skip(image_count * 16);
        skip(8); // checkpoint retention counter
        for (unsigned i = 0; i < 5; ++i) {
            const uint64_t n = integer(); account(n, 1); skip(n);
        }
        return equal ? static_cast<int64_t>(count) : 0;
    }
};
} // namespace

namespace {
// An asset's digest covers every byte, like one SHA-256 over the file, but in a form all cores can compute:
// SHA-256 over a domain tag, the file size, the piece size and the SHA-256 of each 64 MiB piece in order.
constexpr uint64_t kAssetPiece = uint64_t(64) << 20;
constexpr size_t kAssetRead = size_t(8) << 20;

ConversationDigest hash_asset_piece(const std::filesystem::path& path, uint64_t offset, uint64_t count,
                                    std::vector<char>& buffer) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open identity asset: " + path.string());
    if (offset && !f.seekg(std::streamoff(offset))) throw std::runtime_error("cannot read identity asset: " + path.string());
    Digest piece;
    while (count) {
        const size_t n = size_t(std::min<uint64_t>(count, buffer.size()));
        if (!f.read(buffer.data(), std::streamsize(n))) throw std::runtime_error("cannot read identity asset: " + path.string());
        piece.update(buffer.data(), n);
        count -= n;
    }
    return piece.finish();
}

ConversationDigest asset_digest(uint64_t size, const std::vector<ConversationDigest>& pieces) {
    Digest tree;
    const std::string domain = "strata-asset-pieces-v1";
    tree.update(domain.data(), domain.size());
    for (const uint64_t n : {size, kAssetPiece}) { const auto b = little(n); tree.update(b.data(), b.size()); }
    for (const auto& d : pieces) tree.update(d.data(), d.size());
    return tree.finish();
}
} // namespace

bool conversation_identity(const std::vector<ConversationAsset>& assets, const ConversationSettings& settings,
                           ConversationIdentity& identity, std::string& error) {
    try {
        ConversationIdentity result;
        auto field = [](ConversationIdentity& into, const std::string& name, const ConversationDigest& digest) {
            if (into.fields.size() == 256 || name.empty() || name.size() > 128 ||
                name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-") != std::string::npos)
                throw std::runtime_error("invalid or excessive snapshot identity fields");
            if (std::any_of(into.fields.begin(), into.fields.end(), [&](const auto& f) { return f.name == name; }))
                throw std::runtime_error("duplicate snapshot identity field: " + name);
            into.fields.push_back({name, digest});
        };
        for (const auto& [name, value] : settings) {
            Digest hash; hash.update(value.data(), value.size());
            field(result, "runtime/" + name, hash.finish());
        }
        // Hash each complete file once per startup, even when several roles
        // share a GGUF. Never trust a file-stat-only cache across processes.
        // The roles' names are checked before any file is read. Then the distinct files' 64 MiB pieces are hashed
        // on all but one of the CPU's threads, so a model reads at the drive's speed, even from a single file.
        {
            ConversationIdentity names = result;
            for (const auto& asset : assets) field(names, "asset/" + asset.role, {});
        }
        std::vector<std::filesystem::path> files;   // distinct, in the order the assets name them
        std::vector<size_t> file_of;                 // per asset
        for (const auto& asset : assets) {
            const auto path = std::filesystem::canonical(asset.path);
            const auto at = std::find(files.begin(), files.end(), path);
            file_of.push_back(size_t(at - files.begin()));
            if (at == files.end()) files.push_back(path);
        }
        struct Piece { size_t file; uint64_t offset, count; };
        std::vector<uint64_t> sizes(files.size());
        std::vector<std::vector<ConversationDigest>> pieces(files.size());
        std::vector<Piece> work_list;
        for (size_t i = 0; i < files.size(); ++i) {
            sizes[i] = std::filesystem::file_size(files[i]);
            pieces[i].resize(size_t((sizes[i] + kAssetPiece - 1) / kAssetPiece));
            for (uint64_t at = 0; at < sizes[i]; at += kAssetPiece)
                work_list.push_back({i, at, std::min(kAssetPiece, sizes[i] - at)});
        }
        std::vector<std::exception_ptr> failures(files.size());
        std::mutex failure_lock;
        std::atomic<size_t> taken{0};
        std::atomic<bool> failed{false};
        const auto work = [&]() noexcept {
            std::vector<char> buffer;
            for (size_t k; (k = taken.fetch_add(1)) < work_list.size();) {
                const Piece& piece = work_list[k];
                if (failed.load(std::memory_order_relaxed)) continue;
                try {
                    if (buffer.empty()) buffer.resize(kAssetRead);
                    pieces[piece.file][size_t(piece.offset / kAssetPiece)] =
                        hash_asset_piece(files[piece.file], piece.offset, piece.count, buffer);
                } catch (...) {
                    const std::lock_guard<std::mutex> hold(failure_lock);
                    if (!failures[piece.file]) failures[piece.file] = std::current_exception();
                    failed.store(true, std::memory_order_relaxed);
                }
            }
        };
        {
            const unsigned cpus = std::thread::hardware_concurrency();
            const size_t count = std::max<size_t>(1, std::min<size_t>(work_list.size(), cpus > 1 ? cpus - 1 : 1));
            std::vector<std::thread> workers;
            try {
                for (size_t w = 1; w < count; ++w) workers.emplace_back(work);
            } catch (const std::system_error&) {}   // fewer threads: this one takes what is left
            work();
            for (auto& t : workers) t.join();
        }
        for (const auto& failure : failures)
            if (failure) std::rethrow_exception(failure);   // the first file the assets name that failed
        std::vector<ConversationDigest> digests(files.size());
        for (size_t i = 0; i < files.size(); ++i) digests[i] = asset_digest(sizes[i], pieces[i]);
        for (size_t a = 0; a < assets.size(); ++a) field(result, "asset/" + assets[a].role, digests[file_of[a]]);
        Digest hash;
        const std::string domain = "strata-conversation-state-v2";
        hash.update(domain.data(), domain.size());
        for (const auto& f : result.fields) {
            const auto n = little(f.name.size()); hash.update(n.data(), n.size());
            hash.update(f.name.data(), f.name.size()); hash.update(f.digest.data(), f.digest.size());
        }
        result.digest = hash.finish();
        identity = std::move(result);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

namespace {
uint64_t identity_bytes(const ConversationIdentity& identity) {
    if (identity.fields.size() > 256) throw std::runtime_error("too many snapshot identity fields");
    uint64_t bytes = 32 + 8;
    for (const auto& field : identity.fields) {
        if (field.name.empty() || field.name.size() > 128)
            throw std::runtime_error("invalid snapshot identity field name");
        add(bytes, 8 + field.name.size() + field.digest.size());
    }
    return bytes;
}
void write_identity(Writer& w, const ConversationIdentity& identity) {
    w.bytes(identity.digest.data(), identity.digest.size()); w.integer(identity.fields.size());
    for (const auto& f : identity.fields) {
        w.integer(f.name.size()); w.bytes(f.name.data(), f.name.size()); w.bytes(f.digest.data(), f.digest.size());
    }
}
template<class Input> void check_identity(Input& input, const ConversationIdentity& expected) {
    ConversationDigest stored;
    input.bytes(stored.data(), stored.size());
    const auto count = input.integer();
    if (count > 256) throw std::runtime_error("invalid snapshot identity field count");
    // Compare in bounded stack storage before admitting or decoding any state.
    for (size_t i = 0; i < count; ++i) {
        const auto length = input.integer();
        if (!length || length > 128) throw std::runtime_error("invalid snapshot identity field name");
        std::array<char, 128> name{}; ConversationDigest digest;
        input.bytes(name.data(), size_t(length)); input.bytes(digest.data(), digest.size());
        if (i >= expected.fields.size()) throw std::runtime_error("snapshot identity differs: extra field");
        const auto& field = expected.fields[i];
        if (field.name.size() != length || !std::equal(field.name.begin(), field.name.end(), name.begin()) || digest != field.digest)
            throw std::runtime_error("snapshot identity differs: " + field.name);
    }
    if (count < expected.fields.size()) throw std::runtime_error("snapshot identity differs: " + expected.fields[count].name);
    if (stored != expected.digest) throw std::runtime_error("snapshot identity differs: digest");
}
} // namespace

bool conversation_file_write(std::ostream& stream, const SavedConversation& image,
                             const ConversationIdentity& identity, std::string& error, ConversationIoProgress progress) {
    try {
        const auto bound = allocation_bound(image);
        identity_bytes(identity);
        Writer w{stream, {}, {progress}};
        w.bytes(magic.data(), magic.size()); write_identity(w, identity); w.integer(bound);
        for (int64_t n : image.geometry) w.integer(std::bit_cast<uint64_t>(n));
        w.integer(std::bit_cast<uint64_t>(image.layer_lo)); w.integer(std::bit_cast<uint64_t>(image.layer_hi));
        w.integer(image.cvec ? 1 : 0);
        w.checkpoint(image.live);
        w.integer(image.checkpoints.size());
        for (const auto& c : image.checkpoints) w.checkpoint(c);
        w.integer(image.kv.size());
        for (const auto& kv : image.kv) {
            w.integer(static_cast<uint64_t>(kv.format));
            for (int64_t n : {kv.cells, kv.heads, kv.head_dim, kv.page_size, kv.pooled_rows, kv.idx_dim})
                w.integer(std::bit_cast<uint64_t>(n));
            for (const auto* v : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) w.blob(*v);
        }
        const auto sum = w.digest.finish();
        if (!stream.write(reinterpret_cast<const char*>(sum.data()), sum.size()))
            throw std::runtime_error("snapshot footer write failed");
        w.progress.finish();
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool conversation_file_size(const SavedConversation& image, const ConversationIdentity& identity, uint64_t& bytes, std::string& error) {
    try {
        // Header, geometry, steering, checkpoint/KV counts and integrity footer.
        uint64_t total = 8 + identity_bytes(identity) + 8 + 18 * 8 + 16 + 8 + 8 + 8 + 32;
        auto checkpoint = [&](const ConversationCheckpoint& c) {
            if (!c.stage_parts.empty()) throw std::runtime_error("layer-split snapshots are unsupported");
            add(total, 8 * 8); // token/image counts, retention counter, five blob lengths
            add(total, c.ids.size(), 4); add(total, c.imgs.size(), 16);
            for (const auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) add(total, v->size());
        };
        checkpoint(image.live);
        for (const auto& c : image.checkpoints) checkpoint(c);
        for (const auto& kv : image.kv) {
            add(total, 12 * 8); // format, six dimensions, five blob lengths
            for (const auto* v : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) add(total, v->size());
        }
        bytes = total;
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

namespace {
bool read_image(std::istream& stream, const std::filesystem::path* source, unsigned threads,
                const ConversationIdentity& identity, uint64_t staging_limit, std::optional<uint64_t> available,
                uint64_t floor, SavedConversation& output, std::string& error, ConversationIoProgress progress,
                bool* damaged) {
    bool admitted = false;   // past this point a failure is the file's, not the reader's circumstances
    if (damaged) *damaged = false;
    try {
        Reader r{stream, {}, 0, {progress}, source, threads};
        std::array<uint8_t, 8> tag;
        r.bytes(tag.data(), tag.size());
        if (tag != magic) throw std::runtime_error("unsupported conversation snapshot format");
        check_identity(r, identity);
        const uint64_t bound = r.integer();
        if (bound < kConversationFileWorkspace || bound > staging_limit ||
            !core::conversation_memory_admit(available, bound, floor))
            throw std::runtime_error("snapshot staging admission denied");
        r.remaining = bound - kConversationFileWorkspace;
        admitted = true;
        SavedConversation image;
        for (auto& n : image.geometry) n = r.signed_integer();
        image.layer_lo = r.signed_integer(); image.layer_hi = r.signed_integer();
        const auto cvec = r.integer();
        if (cvec > 1) throw std::runtime_error("invalid snapshot steering state");
        image.cvec = cvec != 0;
        r.checkpoint(image.live);
        r.allocate(image.checkpoints, r.integer(), checkpoint_overhead);
        for (auto& c : image.checkpoints) r.checkpoint(c);
        r.allocate(image.kv, r.integer(), kv_overhead);
        for (auto& kv : image.kv) {
            const auto format = r.integer();
            if (format > 3) throw std::runtime_error("unsupported snapshot KV format");
            kv.format = static_cast<int>(format);
            kv.cells = r.signed_integer(); kv.heads = r.signed_integer(); kv.head_dim = r.signed_integer();
            kv.page_size = r.signed_integer(); kv.pooled_rows = r.signed_integer(); kv.idx_dim = r.signed_integer();
            for (auto* v : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) r.blob(*v);
        }
        ConversationDigest footer;
        if (!stream.read(reinterpret_cast<char*>(footer.data()), footer.size()) || footer != r.digest.finish())
            throw std::runtime_error("snapshot integrity check failed");
        if (stream.peek() != std::char_traits<char>::eof() || stream.bad())
            throw std::runtime_error("snapshot has trailing data or read error");
        output = std::move(image);
        r.progress.finish();
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        if (damaged) *damaged = admitted && dynamic_cast<const std::bad_alloc*>(&e) == nullptr;
        return false;
    }
}
} // namespace

bool conversation_file_read(std::istream& stream, const ConversationIdentity& identity,
                            uint64_t staging_limit, std::optional<uint64_t> available, uint64_t floor,
                            SavedConversation& output, std::string& error, ConversationIoProgress progress,
                            bool* damaged) {
    return read_image(stream, nullptr, 1, identity, staging_limit, available, floor, output, error, progress, damaged);
}

bool conversation_file_read(const std::filesystem::path& path, unsigned threads, const ConversationIdentity& identity,
                            uint64_t staging_limit, std::optional<uint64_t> available, uint64_t floor,
                            SavedConversation& output, std::string& error, ConversationIoProgress progress,
                            bool* damaged) {
    if (damaged) *damaged = false;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { error = "cannot open snapshot"; return false; }
    return read_image(stream, &path, threads, identity, staging_limit, available, floor, output, error, progress, damaged);
}

bool conversation_file_match(std::istream& stream, const ConversationIdentity& identity,
                             uint64_t staging_limit, const std::vector<int64_t>& prompt,
                             const std::vector<ConversationImageKey>& images, bool cvec,
                             ConversationFileMatch& match, std::string& error, ConversationIoProgress progress) {
    try {
        if (!stream.seekg(0, std::ios::end)) throw std::runtime_error("snapshot metadata requires seekable input");
        const auto end = stream.tellg();
        if (end < std::streampos(0) || !stream.seekg(0)) throw std::runtime_error("cannot measure snapshot file");
        Probe p{stream, static_cast<uint64_t>(end), 0, {progress}};
        std::array<uint8_t, 8> tag;
        p.bytes(tag.data(), tag.size());
        if (tag != magic) throw std::runtime_error("unsupported conversation snapshot format");
        check_identity(p, identity);
        const uint64_t bound = p.integer();
        if (bound < kConversationFileWorkspace || bound > staging_limit)
            throw std::runtime_error("snapshot staging admission denied");
        p.allocation_remaining = bound - kConversationFileWorkspace;
        p.skip(18 * 8 + 16); // geometry and session layer range is validated by the shared core after selection
        const auto steering = p.integer();
        if (steering > 1) throw std::runtime_error("invalid snapshot steering state");
        ConversationFileMatch found{0, false, bound};
        const int64_t live = p.checkpoint(prompt, images);
        if (live > 0) found = {live, true, bound};
        const uint64_t checkpoints = p.integer(); p.account(checkpoints, checkpoint_overhead);
        for (uint64_t i = 0; i < checkpoints; ++i) {
            const int64_t n = p.checkpoint(prompt, images);
            if (n > found.tokens) found = {n, false, bound};
        }
        if (bool(steering) != cvec) found.tokens = 0;
        match = found;
        p.progress.finish();
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
} // namespace strata::platform

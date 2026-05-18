#include "frame_source.hpp"

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace io {

// ─── EvenlySpacedSource ──────────────────────────────────────────────────────

EvenlySpacedSource::EvenlySpacedSource(int n_frames) : n_frames_(n_frames) {}

double EvenlySpacedSource::probe_duration(const fs::path& video) const {
    // ffprobe -v quiet -show_entries format=duration -of csv=p=0 <file>
    std::string cmd = "ffprobe -v quiet -show_entries format=duration"
                      " -of csv=p=0 \"" + video.string() + "\" 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return 0.0;

    char buf[64] = {};
    if (fgets(buf, sizeof(buf), pipe) == nullptr) {
        pclose(pipe);
        return 0.0;
    }
    pclose(pipe);

    try {
        return std::stod(buf);
    } catch (...) {
        return 0.0;
    }
}

std::vector<fs::path> EvenlySpacedSource::extract(
    const fs::path& video,
    const fs::path& out_dir) const
{
    double duration = probe_duration(video);
    if (duration <= 0.0) return {};

    double interval = duration / n_frames_;
    std::vector<fs::path> frames;
    frames.reserve(n_frames_);

    for (int i = 0; i < n_frames_; ++i) {
        double t = interval * (i + 0.5);
        fs::path out_path = out_dir / ("frame_" + std::to_string(i) + ".png");

        // One ffmpeg call per frame: seek to t, extract exactly 1 frame.
        // -loglevel quiet suppresses all ffmpeg output.
        std::ostringstream cmd;
        cmd << "ffmpeg -loglevel quiet"
            << " -ss " << t
            << " -i \"" << video.string() << "\""
            << " -frames:v 1 -q:v 2"
            << " \"" << out_path.string() << "\""
            << " 2>/dev/null";

        int rc = std::system(cmd.str().c_str());
        if (rc == 0 && fs::exists(out_path)) {
            frames.push_back(out_path);
        }
    }

    return frames;
}

// ─── Factory ─────────────────────────────────────────────────────────────────

std::unique_ptr<FrameSource> make_frame_source(const std::string& strategy,
                                               int n_frames)
{
    // Currently only one strategy; add more here as they are implemented.
    (void)strategy; // strategy param reserved for future strategies
    return std::make_unique<EvenlySpacedSource>(n_frames);
}

} // namespace io

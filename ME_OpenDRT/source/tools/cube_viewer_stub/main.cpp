#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <GLFW/glfw3.h>

#include "OpenDRTParams.h"
#include "OpenDRTPresets.h"
#include "OpenDRTProcessor.h"

namespace {
std::atomic<bool> gRun{true};
std::atomic<bool> gConnected{false};
std::atomic<bool> gBringToFront{false};
std::atomic<int> gWindowVisible{1};
std::atomic<int> gWindowIconified{0};
std::atomic<int> gWindowFocused{1};

void onSignal(int) {
  gRun.store(false);
}

std::string pipeName() {
  const char* env = std::getenv("ME_OPENDRT_CUBE_VIEWER_PIPE");
  if (env && env[0] != '\0') return std::string(env);
#if defined(_WIN32)
  return "\\\\.\\pipe\\ME_OpenDRT_CubeViewer";
#else
  return "/tmp/me_opendrt_cube_viewer.sock";
#endif
}

std::string viewerLogPath() {
#if defined(_WIN32)
  return std::string();
#elif defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (!home || home[0] == '\0') return "/tmp/ME_OpenDRT_CubeViewer.log";
  return std::string(home) + "/Library/Logs/ME_OpenDRT_CubeViewer.log";
#else
  const char* home = std::getenv("HOME");
  if (!home || home[0] == '\0') return "/tmp/ME_OpenDRT_CubeViewer.log";
  return std::string(home) + "/.cache/ME_OpenDRT_CubeViewer.log";
#endif
}

void logViewerEvent(const std::string& msg) {
#if !defined(_WIN32)
  const std::string path = viewerLogPath();
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) return;
  std::fprintf(f, "[ME_OpenDRT_CubeViewer] %s\n", msg.c_str());
  std::fclose(f);
#else
  (void)msg;
#endif
}

const char* kViewerVersionString = "v1.2.9c";

#if !defined(_WIN32)
bool sendAllSocket(int fd, const char* data, size_t size) {
  if (fd < 0 || data == nullptr) return false;
  size_t totalSent = 0;
  while (totalSent < size) {
    const ssize_t sent = ::send(fd, data + totalSent, size - totalSent, 0);
    if (sent <= 0) {
      logViewerEvent(std::string("socket send failed after ") + std::to_string(totalSent) + "/" +
                     std::to_string(size) + " bytes: errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")");
      return false;
    }
    totalSent += static_cast<size_t>(sent);
  }
  return true;
}
#endif

struct CameraState {
  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  float qw = 1.0f;
  float distance = 4.35f;
  float panX = 0.0f;
  float panY = 0.12f;
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct Quat {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 1.0f;
};

inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

inline float length3(Vec3 v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Vec3 normalize3(Vec3 v) {
  const float len = length3(v);
  if (len <= 1e-8f) return Vec3{};
  const float inv = 1.0f / len;
  return Vec3{v.x * inv, v.y * inv, v.z * inv};
}

inline Vec3 cross3(Vec3 a, Vec3 b) {
  return Vec3{
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x};
}

inline float dot3(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Quat normalizeQ(Quat q) {
  const float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (len <= 1e-8f) return Quat{};
  const float inv = 1.0f / len;
  return Quat{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

inline Quat mulQ(Quat a, Quat b) {
  return Quat{
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline Quat axisAngleQ(Vec3 axis, float radians) {
  const Vec3 n = normalize3(axis);
  const float h = radians * 0.5f;
  const float s = std::sin(h);
  return normalizeQ(Quat{n.x * s, n.y * s, n.z * s, std::cos(h)});
}

Vec3 mapArcball(double sx, double sy, int width, int height) {
  if (width < 1) width = 1;
  if (height < 1) height = 1;
  const float nx = static_cast<float>((2.0 * sx - static_cast<double>(width)) / static_cast<double>(width));
  const float ny = static_cast<float>((static_cast<double>(height) - 2.0 * sy) / static_cast<double>(height));
  const float d2 = nx * nx + ny * ny;
  if (d2 <= 1.0f) return Vec3{nx, ny, std::sqrt(1.0f - d2)};
  const float inv = 1.0f / std::sqrt(d2);
  return Vec3{nx * inv, ny * inv, 0.0f};
}

void quatToMatrix(Quat q, float out16[16]) {
  q = normalizeQ(q);
  const float xx = q.x * q.x;
  const float yy = q.y * q.y;
  const float zz = q.z * q.z;
  const float xy = q.x * q.y;
  const float xz = q.x * q.z;
  const float yz = q.y * q.z;
  const float wx = q.w * q.x;
  const float wy = q.w * q.y;
  const float wz = q.w * q.z;
  // Column-major for OpenGL fixed-function pipeline.
  out16[0] = 1.0f - 2.0f * (yy + zz);
  out16[1] = 2.0f * (xy + wz);
  out16[2] = 2.0f * (xz - wy);
  out16[3] = 0.0f;
  out16[4] = 2.0f * (xy - wz);
  out16[5] = 1.0f - 2.0f * (xx + zz);
  out16[6] = 2.0f * (yz + wx);
  out16[7] = 0.0f;
  out16[8] = 2.0f * (xz + wy);
  out16[9] = 2.0f * (yz - wx);
  out16[10] = 1.0f - 2.0f * (xx + yy);
  out16[11] = 0.0f;
  out16[12] = 0.0f;
  out16[13] = 0.0f;
  out16[14] = 0.0f;
  out16[15] = 1.0f;
}

void resetCamera(CameraState* cam) {
  if (!cam) return;
  cam->distance = 4.35f;
  cam->panX = 0.0f;
  cam->panY = 0.12f;
  // Preserve previous default orientation: pitch 20 then yaw -45.
  const float deg2rad = 3.14159265358979323846f / 180.0f;
  const Quat qPitch = axisAngleQ(Vec3{1.0f, 0.0f, 0.0f}, 20.0f * deg2rad);
  const Quat qYaw = axisAngleQ(Vec3{0.0f, 1.0f, 0.0f}, -45.0f * deg2rad);
  const Quat q = normalizeQ(mulQ(qPitch, qYaw));
  cam->qx = q.x;
  cam->qy = q.y;
  cam->qz = q.z;
  cam->qw = q.w;
}

struct MeshData {
  int resolution = 33;
  std::string quality = "Medium";
  std::string paramHash;
  bool renderOk = false;
  float maxDelta = 0.0f;
  std::vector<float> pointVerts;
  std::vector<float> pointColors;
};

struct PendingMessage {
  uint64_t seq = 0;
  std::string line;
};

bool senderMatchesCurrent(const std::string& currentSenderId, const std::string& senderId) {
  return currentSenderId.empty() || senderId.empty() || senderId == currentSenderId;
}

std::mutex gMsgMutex;
PendingMessage gPendingParamsMsg;
PendingMessage gPendingCloudMsg;
bool gHasPendingParamsMsg = false;
bool gHasPendingCloudMsg = false;

inline float clamp01(float v) {
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

void mapDisplayColor(float inR, float inG, float inB, float* outR, float* outG, float* outB) {
  float r = clamp01(inR);
  float g = clamp01(inG);
  float b = clamp01(inB);
  // Keep mapping close to source hues to match scope-like display.
  r = std::pow(r, 0.90f);
  g = std::pow(g, 0.90f);
  b = std::pow(b, 0.90f);
  const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
  const float sat = 1.00f;
  r = clamp01(luma + (r - luma) * sat);
  g = clamp01(luma + (g - luma) * sat);
  b = clamp01(luma + (b - luma) * sat);
  *outR = r;
  *outG = g;
  *outB = b;
}

void applyLookValuesToResolved(OpenDRTParams& p, const LookPresetValues& s) {
  p.tn_con = s.tn_con; p.tn_sh = s.tn_sh; p.tn_toe = s.tn_toe; p.tn_off = s.tn_off;
  p.tn_hcon_enable = s.tn_hcon_enable; p.tn_hcon = s.tn_hcon; p.tn_hcon_pv = s.tn_hcon_pv; p.tn_hcon_st = s.tn_hcon_st;
  p.tn_lcon_enable = s.tn_lcon_enable; p.tn_lcon = s.tn_lcon; p.tn_lcon_w = s.tn_lcon_w;
  p.cwp = s.cwp; p.cwp_lm = s.cwp_lm;
  p.rs_sa = s.rs_sa; p.rs_rw = s.rs_rw; p.rs_bw = s.rs_bw;
  p.pt_enable = s.pt_enable; p.pt_lml = s.pt_lml; p.pt_lml_r = s.pt_lml_r; p.pt_lml_g = s.pt_lml_g; p.pt_lml_b = s.pt_lml_b;
  p.pt_lmh = s.pt_lmh; p.pt_lmh_r = s.pt_lmh_r; p.pt_lmh_b = s.pt_lmh_b;
  p.ptl_enable = s.ptl_enable; p.ptl_c = s.ptl_c; p.ptl_m = s.ptl_m; p.ptl_y = s.ptl_y;
  p.ptm_enable = s.ptm_enable; p.ptm_low = s.ptm_low; p.ptm_low_rng = s.ptm_low_rng; p.ptm_low_st = s.ptm_low_st;
  p.ptm_high = s.ptm_high; p.ptm_high_rng = s.ptm_high_rng; p.ptm_high_st = s.ptm_high_st;
  p.brl_enable = s.brl_enable; p.brl = s.brl; p.brl_r = s.brl_r; p.brl_g = s.brl_g; p.brl_b = s.brl_b; p.brl_rng = s.brl_rng; p.brl_st = s.brl_st;
  p.brlp_enable = s.brlp_enable; p.brlp = s.brlp; p.brlp_r = s.brlp_r; p.brlp_g = s.brlp_g; p.brlp_b = s.brlp_b;
  p.hc_enable = s.hc_enable; p.hc_r = s.hc_r; p.hc_r_rng = s.hc_r_rng;
  p.hs_rgb_enable = s.hs_rgb_enable; p.hs_r = s.hs_r; p.hs_r_rng = s.hs_r_rng; p.hs_g = s.hs_g; p.hs_g_rng = s.hs_g_rng; p.hs_b = s.hs_b; p.hs_b_rng = s.hs_b_rng;
  p.hs_cmy_enable = s.hs_cmy_enable; p.hs_c = s.hs_c; p.hs_c_rng = s.hs_c_rng; p.hs_m = s.hs_m; p.hs_m_rng = s.hs_m_rng; p.hs_y = s.hs_y; p.hs_y_rng = s.hs_y_rng;
}

void applyTonescaleValuesToResolved(OpenDRTParams& p, const TonescalePresetValues& t) {
  p.tn_con = t.tn_con; p.tn_sh = t.tn_sh; p.tn_toe = t.tn_toe; p.tn_off = t.tn_off;
  p.tn_hcon_enable = t.tn_hcon_enable; p.tn_hcon = t.tn_hcon; p.tn_hcon_pv = t.tn_hcon_pv; p.tn_hcon_st = t.tn_hcon_st;
  p.tn_lcon_enable = t.tn_lcon_enable; p.tn_lcon = t.tn_lcon; p.tn_lcon_w = t.tn_lcon_w;
}

bool parseLookValues(const std::string& in, LookPresetValues* v) {
  if (!v) return false;
  std::istringstream is(in);
  return static_cast<bool>(
      is >> v->tn_con >> v->tn_sh >> v->tn_toe >> v->tn_off
         >> v->tn_hcon_enable >> v->tn_hcon >> v->tn_hcon_pv >> v->tn_hcon_st
         >> v->tn_lcon_enable >> v->tn_lcon >> v->tn_lcon_w
         >> v->cwp >> v->cwp_lm
         >> v->rs_sa >> v->rs_rw >> v->rs_bw
         >> v->pt_enable
         >> v->pt_lml >> v->pt_lml_r >> v->pt_lml_g >> v->pt_lml_b
         >> v->pt_lmh >> v->pt_lmh_r >> v->pt_lmh_b
         >> v->ptl_enable >> v->ptl_c >> v->ptl_m >> v->ptl_y
         >> v->ptm_enable >> v->ptm_low >> v->ptm_low_rng >> v->ptm_low_st
         >> v->ptm_high >> v->ptm_high_rng >> v->ptm_high_st
         >> v->brl_enable >> v->brl >> v->brl_r >> v->brl_g >> v->brl_b
         >> v->brl_rng >> v->brl_st
         >> v->brlp_enable >> v->brlp >> v->brlp_r >> v->brlp_g >> v->brlp_b
         >> v->hc_enable >> v->hc_r >> v->hc_r_rng
         >> v->hs_rgb_enable >> v->hs_r >> v->hs_r_rng
         >> v->hs_g >> v->hs_g_rng >> v->hs_b >> v->hs_b_rng
         >> v->hs_cmy_enable >> v->hs_c >> v->hs_c_rng >> v->hs_m >> v->hs_m_rng
         >> v->hs_y >> v->hs_y_rng);
}

bool parseTonescaleValues(const std::string& in, TonescalePresetValues* v) {
  if (!v) return false;
  std::istringstream is(in);
  return static_cast<bool>(
      is >> v->tn_con >> v->tn_sh >> v->tn_toe >> v->tn_off
         >> v->tn_hcon_enable >> v->tn_hcon >> v->tn_hcon_pv >> v->tn_hcon_st
         >> v->tn_lcon_enable >> v->tn_lcon >> v->tn_lcon_w);
}

bool extractJsonNumber(const std::string& src, const char* key, double* out) {
  if (!out) return false;
  const std::string pat = std::string("\"") + key + "\":";
  const size_t p = src.find(pat);
  if (p == std::string::npos) return false;
  const char* begin = src.c_str() + p + pat.size();
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) return false;
  *out = v;
  return true;
}

bool extractJsonInt(const std::string& src, const char* key, int* out) {
  double d = 0.0;
  if (!extractJsonNumber(src, key, &d)) return false;
  *out = static_cast<int>(d);
  return true;
}

bool extractJsonString(const std::string& src, const char* key, std::string* out) {
  if (!out) return false;
  const std::string pat = std::string("\"") + key + "\":\"";
  const size_t p = src.find(pat);
  if (p == std::string::npos) return false;
  size_t i = p + pat.size();
  std::string v;
  while (i < src.size()) {
    char c = src[i++];
    if (c == '\\' && i < src.size()) {
      const char e = src[i++];
      switch (e) {
        case 'n': v.push_back('\n'); break;
        case 'r': v.push_back('\r'); break;
        case 't': v.push_back('\t'); break;
        case '\\': v.push_back('\\'); break;
        case '"': v.push_back('"'); break;
        default: v.push_back(e); break;
      }
      continue;
    }
    if (c == '"') {
      *out = v;
      return true;
    }
    v.push_back(c);
  }
  return false;
}

struct ResolvedPayload {
  uint64_t seq = 0;
  std::string senderId;
  int resolution = 33;
  int in_gamut = 14;
  int in_oetf = 1;
  int lookPreset = 0;
  int tonescalePreset = 0;
  int creativeWhitePreset = 0;
  int displayEncodingPreset = 0;
  float tn_Lp = 100.0f;
  float tn_Lg = 10.0f;
  float tn_gb = 0.13f;
  float pt_hdr = 0.5f;
  int crv_enable = 0;
  int clamp = 1;
  int tn_su = 1;
  int display_gamut = 0;
  int eotf = 2;
  std::string sourceMode = "identity";
  int alwaysOnTop = 0;
  std::string quality = "Medium";
  std::string paramHash;
  std::string lookPayload;
  std::string tonescalePayload;
};

struct InputCloudPayload {
  uint64_t seq = 0;
  std::string senderId;
  int resolution = 33;
  std::string quality = "Medium";
  std::string paramHash;
  std::string points;
};

bool parseParamsMessage(const std::string& msg, ResolvedPayload* out) {
  if (!out) return false;
  std::string type;
  if (!extractJsonString(msg, "type", &type)) return false;
  if (type != "params_snapshot" && type != "params_delta") return false;

  double seqD = 0.0;
  if (!extractJsonNumber(msg, "seq", &seqD)) return false;
  out->seq = static_cast<uint64_t>(seqD < 0.0 ? 0.0 : seqD);
  extractJsonString(msg, "senderId", &out->senderId);

  int resolution = 33;
  if (extractJsonInt(msg, "resolution", &resolution)) out->resolution = resolution;
  extractJsonString(msg, "quality", &out->quality);
  extractJsonString(msg, "sourceMode", &out->sourceMode);
  extractJsonInt(msg, "alwaysOnTop", &out->alwaysOnTop);
  extractJsonString(msg, "paramHash", &out->paramHash);
  extractJsonInt(msg, "in_gamut", &out->in_gamut);
  extractJsonInt(msg, "in_oetf", &out->in_oetf);
  extractJsonInt(msg, "lookPreset", &out->lookPreset);
  extractJsonInt(msg, "tonescalePreset", &out->tonescalePreset);
  extractJsonInt(msg, "creativeWhitePreset", &out->creativeWhitePreset);
  extractJsonInt(msg, "displayEncodingPreset", &out->displayEncodingPreset);
  double f = 0.0;
  if (extractJsonNumber(msg, "tn_Lp", &f)) out->tn_Lp = static_cast<float>(f);
  if (extractJsonNumber(msg, "tn_Lg", &f)) out->tn_Lg = static_cast<float>(f);
  if (extractJsonNumber(msg, "tn_gb", &f)) out->tn_gb = static_cast<float>(f);
  if (extractJsonNumber(msg, "pt_hdr", &f)) out->pt_hdr = static_cast<float>(f);
  extractJsonInt(msg, "crv_enable", &out->crv_enable);
  extractJsonInt(msg, "clamp", &out->clamp);
  extractJsonInt(msg, "tn_su", &out->tn_su);
  extractJsonInt(msg, "display_gamut", &out->display_gamut);
  extractJsonInt(msg, "eotf", &out->eotf);
  extractJsonString(msg, "lookPayload", &out->lookPayload);
  extractJsonString(msg, "tonescalePayload", &out->tonescalePayload);
  return true;
}

bool parseInputCloudMessage(const std::string& msg, InputCloudPayload* out) {
  if (!out) return false;
  std::string type;
  if (!extractJsonString(msg, "type", &type) || type != "input_cloud") return false;
  double seqD = 0.0;
  if (!extractJsonNumber(msg, "seq", &seqD)) return false;
  out->seq = static_cast<uint64_t>(seqD < 0.0 ? 0.0 : seqD);
  extractJsonString(msg, "senderId", &out->senderId);
  int resolution = 33;
  if (extractJsonInt(msg, "resolution", &resolution)) out->resolution = resolution;
  extractJsonString(msg, "quality", &out->quality);
  extractJsonString(msg, "paramHash", &out->paramHash);
  if (!extractJsonString(msg, "points", &out->points)) return false;
  return true;
}

std::string heartbeatAckJson() {
  const int visible = gWindowVisible.load(std::memory_order_relaxed) ? 1 : 0;
  const int minimized = gWindowIconified.load(std::memory_order_relaxed) ? 1 : 0;
  const int focused = gWindowFocused.load(std::memory_order_relaxed) ? 1 : 0;
  const int active = (visible != 0 && minimized == 0) ? 1 : 0;
  std::ostringstream os;
  os << "{\"type\":\"heartbeat_ack\",\"active\":" << active << ",\"visible\":" << visible
     << ",\"minimized\":" << minimized << ",\"focused\":" << focused << "}";
  return os.str();
}

std::string handleIncomingLine(const std::string& line) {
  if (line.empty()) return std::string();
  if (line.find("\"type\":\"hello\"") != std::string::npos || line.find("\"type\":\"open_session\"") != std::string::npos) {
    gConnected.store(true);
    logViewerEvent("Received session hello/open_session.");
    return std::string();
  }
  if (line.find("\"type\":\"close_session\"") != std::string::npos) {
    logViewerEvent("Received close_session.");
    gRun.store(false);
    return std::string();
  }
  if (line.find("\"type\":\"heartbeat\"") != std::string::npos) {
    gConnected.store(true);
    logViewerEvent("Received heartbeat.");
    return heartbeatAckJson();
  }
  if (line.find("\"type\":\"bring_to_front\"") != std::string::npos) {
    gBringToFront.store(true);
    gConnected.store(true);
    logViewerEvent("Received bring_to_front.");
    return std::string();
  }
  ResolvedPayload payload{};
  InputCloudPayload cloud{};
  const bool isParams = parseParamsMessage(line, &payload);
  const bool isCloud = isParams ? false : parseInputCloudMessage(line, &cloud);
  if (!isParams && !isCloud) return std::string();
  std::lock_guard<std::mutex> lock(gMsgMutex);
  if (isParams) {
    std::ostringstream os;
    os << "Queued params message: sender=" << payload.senderId
       << " seq=" << payload.seq
       << " sourceMode=" << payload.sourceMode
       << " quality=" << payload.quality
       << " res=" << payload.resolution;
    logViewerEvent(os.str());
    gPendingParamsMsg.seq = payload.seq;
    gPendingParamsMsg.line = line;
    gHasPendingParamsMsg = true;
  } else {
    std::ostringstream os;
    os << "Queued input cloud message: sender=" << cloud.senderId
       << " seq=" << cloud.seq
       << " quality=" << cloud.quality
       << " res=" << cloud.resolution
       << " pointBytes=" << cloud.points.size();
    logViewerEvent(os.str());
    gPendingCloudMsg.seq = cloud.seq;
    gPendingCloudMsg.line = line;
    gHasPendingCloudMsg = true;
  }
  gConnected.store(true);
  return std::string();
}

OpenDRTParams buildResolvedParams(const ResolvedPayload& rp) {
  OpenDRTParams p{};
  p.in_gamut = rp.in_gamut;
  p.in_oetf = rp.in_oetf;
  p.tn_Lp = rp.tn_Lp;
  p.tn_Lg = rp.tn_Lg;
  p.tn_gb = rp.tn_gb;
  p.pt_hdr = rp.pt_hdr;
  p.crv_enable = rp.crv_enable;
  p.clamp = rp.clamp;
  p.tn_su = rp.tn_su;
  p.display_gamut = rp.display_gamut;
  p.eotf = rp.eotf;

  applyLookPresetToResolved(p, rp.lookPreset);
  applyDisplayEncodingPreset(p, rp.displayEncodingPreset);
  if (rp.creativeWhitePreset > 0) p.cwp = rp.creativeWhitePreset - 1;

  LookPresetValues lookVals{};
  if (!rp.lookPayload.empty() && parseLookValues(rp.lookPayload, &lookVals)) {
    applyLookValuesToResolved(p, lookVals);
  }

  if (rp.tonescalePreset > 0) {
    applyTonescalePresetToResolved(p, rp.tonescalePreset);
  }
  TonescalePresetValues toneVals{};
  if (!rp.tonescalePayload.empty() && parseTonescaleValues(rp.tonescalePayload, &toneVals)) {
    applyTonescaleValuesToResolved(p, toneVals);
  }
  return p;
}

void buildCubeData(const ResolvedPayload& payload, MeshData* out) {
  if (!out) return;
  const int res = (payload.resolution <= 25) ? 25 : (payload.resolution <= 41 ? 41 : 57);
  const size_t count = static_cast<size_t>(res) * static_cast<size_t>(res) * static_cast<size_t>(res);

  std::vector<float> src(count * 4u, 1.0f);
  auto idx3 = [res](int x, int y, int z) -> size_t {
    return (static_cast<size_t>(z) * static_cast<size_t>(res) + static_cast<size_t>(y)) * static_cast<size_t>(res) +
           static_cast<size_t>(x);
  };
  const float denom = static_cast<float>(res - 1);
  for (int z = 0; z < res; ++z) {
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        const size_t i = idx3(x, y, z) * 4u;
        src[i + 0] = static_cast<float>(x) / denom;
        src[i + 1] = static_cast<float>(y) / denom;
        src[i + 2] = static_cast<float>(z) / denom;
      }
    }
  }

  std::vector<float> dst(count * 4u, 1.0f);
  OpenDRTProcessor proc(buildResolvedParams(payload));
  const bool renderOk = proc.render(src.data(), dst.data(), static_cast<int>(count), 1, true, true);
  if (!renderOk) {
    dst = src;
  }

  MeshData mesh{};
  mesh.resolution = res;
  mesh.quality = payload.quality;
  mesh.paramHash = payload.paramHash;
  mesh.renderOk = renderOk;
  float maxDelta = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    const size_t si = i * 4u;
    const float dr = std::fabs(dst[si + 0] - src[si + 0]);
    const float dg = std::fabs(dst[si + 1] - src[si + 1]);
    const float db = std::fabs(dst[si + 2] - src[si + 2]);
    if (dr > maxDelta) maxDelta = dr;
    if (dg > maxDelta) maxDelta = dg;
    if (db > maxDelta) maxDelta = db;
  }
  mesh.maxDelta = maxDelta;
  const int interiorStep = (res <= 25) ? 2 : (res <= 41 ? 2 : 3);
  mesh.pointVerts.reserve((count / static_cast<size_t>(interiorStep * interiorStep * interiorStep) + 1u) * 3u);
  mesh.pointColors.reserve(mesh.pointVerts.capacity());
  for (int z = 0; z < res; ++z) {
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        const bool onBoundary = (x == 0 || y == 0 || z == 0 || x == res - 1 || y == res - 1 || z == res - 1);
        if (!onBoundary) {
          if ((x % interiorStep) != 0 || (y % interiorStep) != 0 || (z % interiorStep) != 0) continue;
        }
        const size_t i = idx3(x, y, z);
        const size_t si = i * 4u;
        const float rx = dst[si + 0];
        const float ry = dst[si + 1];
        const float rz = dst[si + 2];
        mesh.pointVerts.push_back(rx * 2.0f - 1.0f);
        mesh.pointVerts.push_back(ry * 2.0f - 1.0f);
        mesh.pointVerts.push_back(rz * 2.0f - 1.0f);
        // Blend source hue with transformed value to better communicate shape and look intent.
        const float mixR = src[si + 0] * 0.86f + clamp01(rx) * 0.14f;
        const float mixG = src[si + 1] * 0.86f + clamp01(ry) * 0.14f;
        const float mixB = src[si + 2] * 0.86f + clamp01(rz) * 0.14f;
        float cr = 0.0f, cg = 0.0f, cb = 0.0f;
        mapDisplayColor(mixR, mixG, mixB, &cr, &cg, &cb);
        mesh.pointColors.push_back(cr);
        mesh.pointColors.push_back(cg);
        mesh.pointColors.push_back(cb);
      }
    }
  }
  *out = std::move(mesh);
}

bool buildInputCloudMesh(const InputCloudPayload& payload, MeshData* out) {
  if (!out) return false;
  std::istringstream is(payload.points);
  MeshData mesh{};
  mesh.resolution = (payload.resolution <= 25) ? 25 : (payload.resolution <= 41 ? 41 : 57);
  mesh.quality = payload.quality;
  mesh.paramHash = payload.paramHash;
  mesh.renderOk = true;
  mesh.maxDelta = 0.0f;
  float sr = 0.0f, sg = 0.0f, sb = 0.0f, dr = 0.0f, dg = 0.0f, db = 0.0f;
  while (is >> sr >> sg >> sb >> dr >> dg >> db) {
    mesh.pointVerts.push_back(dr * 2.0f - 1.0f);
    mesh.pointVerts.push_back(dg * 2.0f - 1.0f);
    mesh.pointVerts.push_back(db * 2.0f - 1.0f);
    const float mixR = clamp01(sr);
    const float mixG = clamp01(sg);
    const float mixB = clamp01(sb);
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    mapDisplayColor(mixR, mixG, mixB, &cr, &cg, &cb);
    mesh.pointColors.push_back(cr);
    mesh.pointColors.push_back(cg);
    mesh.pointColors.push_back(cb);
  }
  if (mesh.pointVerts.empty()) return false;
  *out = std::move(mesh);
  return true;
}

void updateProjection(int width, int height) {
  if (width < 1) width = 1;
  if (height < 1) height = 1;
  const double aspect = static_cast<double>(width) / static_cast<double>(height);
  const double fovy = 45.0;
  const double zNear = 0.1;
  const double zFar = 200.0;
  const double ymax = zNear * std::tan(fovy * 0.5 * 3.141592653589793 / 180.0);
  const double xmax = ymax * aspect;
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(-xmax, xmax, -ymax, ymax, zNear, zFar);
  glMatrixMode(GL_MODELVIEW);
}

void drawReferenceFrame() {
  static const float kCubeEdges[] = {
      -1.f,-1.f,-1.f,  1.f,-1.f,-1.f,
       1.f,-1.f,-1.f,  1.f, 1.f,-1.f,
       1.f, 1.f,-1.f, -1.f, 1.f,-1.f,
      -1.f, 1.f,-1.f, -1.f,-1.f,-1.f,
      -1.f,-1.f, 1.f,  1.f,-1.f, 1.f,
       1.f,-1.f, 1.f,  1.f, 1.f, 1.f,
       1.f, 1.f, 1.f, -1.f, 1.f, 1.f,
      -1.f, 1.f, 1.f, -1.f,-1.f, 1.f,
      -1.f,-1.f,-1.f, -1.f,-1.f, 1.f,
       1.f,-1.f,-1.f,  1.f,-1.f, 1.f,
       1.f, 1.f,-1.f,  1.f, 1.f, 1.f,
      -1.f, 1.f,-1.f, -1.f, 1.f, 1.f
  };
  static const float kAxes[] = {
      -1.f,-1.f,-1.f, 1.35f,-1.f,-1.f,
      -1.f,-1.f,-1.f, -1.f,1.35f,-1.f,
      -1.f,-1.f,-1.f, -1.f,-1.f,1.35f
  };
  static const float kNeutralAxis[] = {
      -1.f,-1.f,-1.f, 1.f,1.f,1.f
  };
  glEnableClientState(GL_VERTEX_ARRAY);
  glLineWidth(1.15f);
  glColor4f(0.97f, 0.97f, 0.97f, 0.55f);
  glVertexPointer(3, GL_FLOAT, 0, kCubeEdges);
  glDrawArrays(GL_LINES, 0, 24);

  glLineWidth(1.5f);
  glVertexPointer(3, GL_FLOAT, 0, kAxes);
  glColor4f(1.0f, 0.32f, 0.32f, 0.9f);
  glDrawArrays(GL_LINES, 0, 2);
  glColor4f(0.35f, 1.0f, 0.35f, 0.9f);
  glDrawArrays(GL_LINES, 2, 2);
  glColor4f(0.35f, 0.60f, 1.0f, 0.9f);
  glDrawArrays(GL_LINES, 4, 2);
  glLineWidth(1.2f);
  glVertexPointer(3, GL_FLOAT, 0, kNeutralAxis);
  glColor4f(1.0f, 1.0f, 1.0f, 0.38f);
  glDrawArrays(GL_LINES, 0, 2);
  glDisableClientState(GL_VERTEX_ARRAY);
}

void updateTitle(GLFWwindow* window, const MeshData& mesh, const char* state) {
  std::ostringstream os;
  os << "ME_OpenDRT Cube Viewer | " << state << " | " << mesh.quality << " " << mesh.resolution << "^3";
  os << " | math:" << (mesh.renderOk ? "ok" : "fallback");
  os << " | dMax:" << mesh.maxDelta;
  if (!mesh.paramHash.empty()) os << " | hash " << mesh.paramHash;
  glfwSetWindowTitle(window, os.str().c_str());
}

#if defined(_WIN32)
void ipcServerLoop() {
  const std::string name = pipeName();
  while (gRun.load()) {
    HANDLE hPipe = CreateNamedPipeA(
        name.c_str(),
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1,
        0,
        1 << 16,
        100,
        nullptr);
    if (hPipe == INVALID_HANDLE_VALUE) {
      Sleep(250);
      continue;
    }
    const BOOL connected = ConnectNamedPipe(hPipe, nullptr) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
    if (!connected) {
      CloseHandle(hPipe);
      continue;
    }
    std::string pending;
    char buf[8192];
    DWORD readBytes = 0;
    while (gRun.load() && ReadFile(hPipe, buf, sizeof(buf), &readBytes, nullptr) && readBytes > 0) {
      pending.append(buf, buf + readBytes);
      size_t nl = std::string::npos;
      while ((nl = pending.find('\n')) != std::string::npos) {
        std::string line = pending.substr(0, nl);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string response = handleIncomingLine(line);
        if (!response.empty()) {
          std::string payload = response;
          payload.push_back('\n');
          DWORD written = 0;
          (void)WriteFile(hPipe, payload.data(), static_cast<DWORD>(payload.size()), &written, nullptr);
        }
        pending.erase(0, nl + 1);
      }
    }
    DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);
  }
}

void wakeIpcServer() {
  HANDLE h = CreateFileA(pipeName().c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    DWORD w = 0;
    const char nl = '\n';
    WriteFile(h, &nl, 1, &w, nullptr);
    CloseHandle(h);
  }
}
#else
void ipcServerLoop() {
  const std::string path = pipeName();
  ::unlink(path.c_str());
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    logViewerEvent(std::string("IPC socket() failed: errno=") + std::to_string(errno) + " (" + std::strerror(errno) + ")");
    return;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    logViewerEvent(
        std::string("IPC bind() failed for ") + path + ": errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")");
    ::close(fd);
    return;
  }
  if (::listen(fd, 4) != 0) {
    logViewerEvent(std::string("IPC listen() failed: errno=") + std::to_string(errno) + " (" + std::strerror(errno) + ")");
    ::close(fd);
    ::unlink(path.c_str());
    return;
  }

  while (gRun.load()) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    const int sel = ::select(fd + 1, &rfds, nullptr, nullptr, &tv);
    if (sel <= 0) continue;

    const int client = ::accept(fd, nullptr, nullptr);
    if (client < 0) continue;
    std::string pending;
    char buf[8192];
    while (gRun.load()) {
      const ssize_t n = ::recv(client, buf, sizeof(buf), 0);
      if (n <= 0) break;
      pending.append(buf, buf + n);
      size_t nl = std::string::npos;
      while ((nl = pending.find('\n')) != std::string::npos) {
        std::string line = pending.substr(0, nl);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string response = handleIncomingLine(line);
        if (!response.empty()) {
          std::string payload = response;
          payload.push_back('\n');
          (void)sendAllSocket(client, payload.data(), payload.size());
        }
        pending.erase(0, nl + 1);
      }
    }
    if (!pending.empty()) {
      logViewerEvent(std::string("Connection closed with unterminated payload bytes=") + std::to_string(pending.size()));
    }
    ::close(client);
  }

  ::close(fd);
  ::unlink(path.c_str());
}

void wakeIpcServer() {
  const std::string path = pipeName();
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return;
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    const char nl = '\n';
    (void)sendAllSocket(fd, &nl, 1);
  }
  ::close(fd);
}
#endif

struct AppState {
  CameraState cam;
  bool leftDown = false;
  bool panMode = false;
  bool zoomMode = false;
  bool keepOnTop = false;
  bool appliedTopmost = false;
  std::string currentSourceMode = "identity";
  std::string currentSenderId;
  double lastX = 0.0;
  double lastY = 0.0;
  double lastClick = 0.0;
  float scrollAccum = 0.0f;
};

void onFramebufferSize(GLFWwindow*, int w, int h) {
  updateProjection(w, h);
}

void onWindowClose(GLFWwindow*) {
  gRun.store(false);
}

void onScroll(GLFWwindow* window, double, double yoff) {
  auto* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  const bool shift =
      (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
  const bool ctrl =
      (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
  const float scale = (shift || ctrl) ? 0.35f : 1.0f;
  app->scrollAccum += static_cast<float>(yoff) * scale;
}

void processMouseAndKeys(GLFWwindow* window, AppState* app) {
  if (!app) return;
  const int l = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
  const int m = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);
  const int r = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
  const bool anyDown = (l == GLFW_PRESS || m == GLFW_PRESS || r == GLFW_PRESS);
  const bool shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  double cx = 0.0, cy = 0.0;
  glfwGetCursorPos(window, &cx, &cy);

  if (anyDown && !app->leftDown) {
    app->leftDown = true;
    app->zoomMode = (r == GLFW_PRESS);
    app->panMode = !app->zoomMode && ((m == GLFW_PRESS) || shift);
    app->lastX = cx;
    app->lastY = cy;
    const double now = glfwGetTime();
    if (l == GLFW_PRESS && (now - app->lastClick) < 0.3) {
      resetCamera(&app->cam);
    }
    app->lastClick = now;
  } else if (!anyDown && app->leftDown) {
    app->leftDown = false;
    app->panMode = false;
    app->zoomMode = false;
  }

  if (app->leftDown) {
    const float dx = static_cast<float>(cx - app->lastX);
    const float dy = static_cast<float>(cy - app->lastY);
    app->lastX = cx;
    app->lastY = cy;
    if (app->zoomMode) {
      const float zoomFactor = std::exp(dy * 0.0105f);
      app->cam.distance *= zoomFactor;
      if (app->cam.distance < 0.6f) app->cam.distance = 0.6f;
      if (app->cam.distance > 30.0f) app->cam.distance = 30.0f;
    } else if (app->panMode) {
      const float panScale = 0.0022f * app->cam.distance;
      app->cam.panX += dx * panScale;
      app->cam.panY -= dy * panScale;
    } else {
      int w = 1, h = 1;
      glfwGetWindowSize(window, &w, &h);
      Vec3 v0 = mapArcball(app->lastX - dx, app->lastY - dy, w, h);
      Vec3 v1 = mapArcball(app->lastX, app->lastY, w, h);
      Vec3 axis = cross3(v0, v1);
      const float axisLen = length3(axis);
      float d = dot3(v0, v1);
      d = clampf(d, -1.0f, 1.0f);
      const float angle = std::acos(d);
      if (axisLen > 1e-6f && angle > 1e-6f) {
        const float arcballGain = 1.35f;
        const Quat qDelta = axisAngleQ(axis, angle * arcballGain);
        Quat qCur{app->cam.qx, app->cam.qy, app->cam.qz, app->cam.qw};
        qCur = normalizeQ(mulQ(qDelta, qCur));
        app->cam.qx = qCur.x;
        app->cam.qy = qCur.y;
        app->cam.qz = qCur.z;
        app->cam.qw = qCur.w;
      }
    }
  }

  if (app->scrollAccum != 0.0f) {
    const float delta = app->scrollAccum;
    app->scrollAccum = 0.0f;
    const float factor = std::exp(-delta / 10.0f);
    app->cam.distance *= factor;
    if (app->cam.distance < 0.6f) app->cam.distance = 0.6f;
    if (app->cam.distance > 30.0f) app->cam.distance = 30.0f;
  }

  if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
    resetCamera(&app->cam);
  }
}

int runApp() {
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

#if defined(_WIN32)
  HANDLE singleInstanceMutex = CreateMutexA(nullptr, FALSE, "Global\\ME_OpenDRT_CubeViewer_Singleton");
  if (singleInstanceMutex != nullptr && GetLastError() == ERROR_ALREADY_EXISTS) {
    if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
    return 0;
  }
  _putenv_s("ME_OPENDRT_DISABLE_OPENCL", "0");
  _putenv_s("ME_OPENDRT_FORCE_OPENCL", "1");
#endif

  if (!glfwInit()) {
    logViewerEvent("glfwInit() failed");
#if defined(_WIN32)
    if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
#endif
    return 1;
  }

#if defined(__APPLE__)
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
#endif

  GLFWwindow* window = glfwCreateWindow(864, 560, "ME_OpenDRT Cube Viewer", nullptr, nullptr);
  if (!window) {
    logViewerEvent("glfwCreateWindow() failed");
    glfwTerminate();
#if defined(_WIN32)
    if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
#endif
    return 1;
  }
  logViewerEvent(std::string("Viewer startup ok ") + kViewerVersionString);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) {
    if (const GLFWvidmode* mode = glfwGetVideoMode(monitor)) {
      const int winW = 864;
      const int winH = 560;
      int x = (mode->width - winW) / 2;
      int y = (mode->height - winH) / 2 - 40;
      if (x < 0) x = 0;
      if (y < 0) y = 0;
      glfwSetWindowPos(window, x, y);
    }
  }

  AppState app{};
  resetCamera(&app.cam);
  glfwSetWindowUserPointer(window, &app);
  glfwSetFramebufferSizeCallback(window, onFramebufferSize);
  glfwSetWindowCloseCallback(window, onWindowClose);
  glfwSetScrollCallback(window, onScroll);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_POINT_SMOOTH);
  glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

  int fbW = 0, fbH = 0;
  glfwGetFramebufferSize(window, &fbW, &fbH);
  updateProjection(fbW, fbH);

  std::thread ipcThread(ipcServerLoop);

  MeshData mesh{};
  mesh.resolution = 33;
  mesh.quality = "Medium";
  mesh.renderOk = true;
  mesh.maxDelta = 0.0f;
  uint64_t lastParamsSeq = 0;
  uint64_t lastCloudSeq = 0;
  InputCloudPayload deferredCloud{};
  bool hasDeferredCloud = false;

  while (gRun.load() && !glfwWindowShouldClose(window)) {
    glfwPollEvents();
    gWindowVisible.store(glfwGetWindowAttrib(window, GLFW_VISIBLE) == GLFW_TRUE ? 1 : 0, std::memory_order_relaxed);
    gWindowIconified.store(glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE ? 1 : 0, std::memory_order_relaxed);
    gWindowFocused.store(glfwGetWindowAttrib(window, GLFW_FOCUSED) == GLFW_TRUE ? 1 : 0, std::memory_order_relaxed);
    processMouseAndKeys(window, &app);

    PendingMessage pendingParams{};
    PendingMessage pendingCloud{};
    bool haveParams = false;
    bool haveCloud = false;
    {
      std::lock_guard<std::mutex> lock(gMsgMutex);
      if (gHasPendingParamsMsg) {
        pendingParams = gPendingParamsMsg;
        gHasPendingParamsMsg = false;
        haveParams = true;
      }
      if (gHasPendingCloudMsg) {
        pendingCloud = gPendingCloudMsg;
        gHasPendingCloudMsg = false;
        haveCloud = true;
      }
    }
    if (haveParams) {
      ResolvedPayload rp{};
      try {
        if (parseParamsMessage(pendingParams.line, &rp)) {
          {
            std::ostringstream os;
            os << "Params received: sender=" << rp.senderId
               << " seq=" << rp.seq
               << " sourceMode=" << rp.sourceMode
               << " quality=" << rp.quality
               << " res=" << rp.resolution
               << " hash=" << rp.paramHash;
            logViewerEvent(os.str());
          }
          if (!rp.senderId.empty() && rp.senderId != app.currentSenderId) {
            app.currentSenderId = rp.senderId;
            lastParamsSeq = 0;
            lastCloudSeq = 0;
            hasDeferredCloud = false;
            logViewerEvent("Active sender changed; reset params/cloud sequence tracking.");
          }
          if (rp.seq < lastParamsSeq) {
            logViewerEvent("Ignored stale params sequence.");
          } else {
            lastParamsSeq = rp.seq;
            const std::string prevSourceMode = app.currentSourceMode;
            app.keepOnTop = (rp.alwaysOnTop != 0);
            app.currentSourceMode = rp.sourceMode;
            if (app.currentSourceMode != "input") {
              MeshData nextMesh{};
              buildCubeData(rp, &nextMesh);
              mesh = std::move(nextMesh);
            } else {
              // Keep last displayed mesh until an input_cloud payload arrives.
              // This avoids flicker/blank frames on mode switch and live param deltas.
              mesh.quality = rp.quality;
              mesh.resolution = rp.resolution;
              mesh.paramHash = rp.paramHash;
              mesh.renderOk = true;
              mesh.maxDelta = 0.0f;
              if (prevSourceMode != "input") {
                // Reset pan only when switching modes so stale framing does not hide the cloud.
                app.cam.panX = 0.0f;
                app.cam.panY = 0.0f;
                logViewerEvent("Params switched source mode to input.");
              }
              if (hasDeferredCloud && deferredCloud.seq >= lastCloudSeq &&
                  senderMatchesCurrent(app.currentSenderId, deferredCloud.senderId)) {
                MeshData nextMesh{};
                if (buildInputCloudMesh(deferredCloud, &nextMesh)) {
                  mesh = std::move(nextMesh);
                  lastCloudSeq = deferredCloud.seq;
                  hasDeferredCloud = false;
                  logViewerEvent("Applied deferred input cloud after params switched to input mode.");
                }
              }
            }
          }
        }
      } catch (const std::exception& e) {
        logViewerEvent(std::string("Exception while processing params message: ") + e.what());
      } catch (...) {
        logViewerEvent("Unknown exception while processing params message.");
      }
    }
    if (haveCloud) {
      InputCloudPayload cp{};
      try {
        if (parseInputCloudMessage(pendingCloud.line, &cp)) {
          {
            std::ostringstream os;
            os << "Input cloud received: sender=" << cp.senderId
               << " seq=" << cp.seq
               << " quality=" << cp.quality
               << " res=" << cp.resolution
               << " pointBytes=" << cp.points.size()
               << " sourceMode=" << app.currentSourceMode;
            logViewerEvent(os.str());
          }
          if (!senderMatchesCurrent(app.currentSenderId, cp.senderId)) {
            // Ignore clouds from a different OFX instance than the active sender.
            logViewerEvent("Ignored input cloud from non-active sender.");
          } else if (cp.seq < lastCloudSeq) {
            logViewerEvent("Ignored stale input cloud sequence.");
          } else if (app.currentSourceMode != "input") {
            deferredCloud = cp;
            hasDeferredCloud = true;
            logViewerEvent("Deferred input cloud until params confirm input mode.");
          } else {
            MeshData nextMesh{};
            if (buildInputCloudMesh(cp, &nextMesh)) {
              mesh = std::move(nextMesh);
              lastCloudSeq = cp.seq;
              hasDeferredCloud = false;
              {
                std::ostringstream os;
                os << "Applied input cloud: seq=" << cp.seq
                   << " points=" << (mesh.pointVerts.size() / 3u)
                   << " quality=" << mesh.quality
                   << " hash=" << mesh.paramHash;
                logViewerEvent(os.str());
              }
            } else {
              logViewerEvent("Input cloud payload parsed but mesh build failed.");
            }
          }
        }
      } catch (const std::exception& e) {
        logViewerEvent(std::string("Exception while processing input cloud message: ") + e.what());
      } catch (...) {
        logViewerEvent("Unknown exception while processing input cloud message.");
      }
    }

    if (app.keepOnTop != app.appliedTopmost) {
      glfwSetWindowAttrib(window, GLFW_FLOATING, app.keepOnTop ? GLFW_TRUE : GLFW_FALSE);
      app.appliedTopmost = app.keepOnTop;
    }

    if (gBringToFront.exchange(false)) {
      glfwRestoreWindow(window);
      glfwFocusWindow(window);
    }

    updateTitle(window, mesh, gConnected.load() ? "Connected" : "Disconnected");

    glClearColor(0.08f, 0.08f, 0.09f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(app.cam.panX, app.cam.panY, -app.cam.distance);
    float rotM[16];
    quatToMatrix(Quat{app.cam.qx, app.cam.qy, app.cam.qz, app.cam.qw}, rotM);
    glMultMatrixf(rotM);

    drawReferenceFrame();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    float clampedDist = app.cam.distance;
    if (clampedDist < 0.6f) clampedDist = 0.6f;
    if (clampedDist > 30.0f) clampedDist = 30.0f;
    float basePointSize = 2.5f;
    if (mesh.resolution <= 25) basePointSize = 3.1f;
    else if (mesh.resolution <= 41) basePointSize = 2.7f;
    else basePointSize = 2.3f;
    const float distanceFactor = std::pow(clampedDist / 4.2f, -0.68f);
    const size_t pointCount = mesh.pointVerts.size() / 3u;
    float densityFactor = 1.0f;
    if (pointCount < 8000u) densityFactor = 1.10f;
    else if (pointCount > 50000u) densityFactor = 0.92f;
    // QUICK_TWEAK_POINT_SIZE: adjust this multiplier to make cube sample points smaller/larger.
    const float kPointSizeUserScale = 1.28f;
    float pointSize = basePointSize * distanceFactor * densityFactor * kPointSizeUserScale;
    if (pointSize < 1.2f) pointSize = 1.2f;
    if (pointSize > 5.0f) pointSize = 5.0f;
    glPointSize(pointSize);
    glVertexPointer(3, GL_FLOAT, 0, mesh.pointVerts.empty() ? nullptr : mesh.pointVerts.data());
    glColorPointer(3, GL_FLOAT, 0, mesh.pointColors.empty() ? nullptr : mesh.pointColors.data());
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(mesh.pointVerts.size() / 3u));
    if (!mesh.pointVerts.empty()) {
      // Subtle interior fill pass to improve interior visibility for both identity and input cloud modes.
      glDisable(GL_DEPTH_TEST);
      glDisableClientState(GL_COLOR_ARRAY);
      glColor4f(0.95f, 0.96f, 1.0f, 0.05f);
      glPointSize(pointSize * 0.55f);
      glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(mesh.pointVerts.size() / 3u));
      glEnableClientState(GL_COLOR_ARRAY);
      glEnable(GL_DEPTH_TEST);
    }
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glfwSwapBuffers(window);
  }

  gRun.store(false);
  wakeIpcServer();
  if (ipcThread.joinable()) ipcThread.join();

  glfwDestroyWindow(window);
  glfwTerminate();

#if defined(_WIN32)
  if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
#endif
  return 0;
}

}  // namespace

#if defined(_WIN32)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
  return runApp();
}
#else
int main() {
  return runApp();
}
#endif


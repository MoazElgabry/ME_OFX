#include <cmath>
#include <chrono>
#include <cctype>
#include <cstdlib>
#if !defined(__linux__)
#include <filesystem>
#endif
#include <fstream>
#include <cstdio>
#include <mutex>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(__linux__)
#include <cerrno>
#include <sys/stat.h>
#endif

#include "ofxsImageEffect.h"

#if defined(OFX_SUPPORTS_CUDARENDER)
#include <cuda_runtime.h>
#endif

#include "OpenDRTParams.h"
#include "OpenDRTPresets.h"
#include "OpenDRTProcessor.h"

#define kPluginName "ME_OpenDRT"
#define kPluginGrouping "Moaz Elgabry"
#define kPluginDescription "OpenDRT v1.1.0 by Jed Smith, ported to OFX by Moaz ELgabry"
#define kPluginIdentifier "com.moazelgabry.me_opendrt"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 3

namespace {

// ===== Startup Mode/Logging Switches =====
// These helpers read env toggles once (static cache) to keep render-time behavior deterministic
// and avoid repeated getenv/string parsing in hot paths.
bool perfLogEnabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("ME_OPENDRT_PERF_LOG");
    if (v == nullptr || v[0] == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

bool forceStageCopyEnabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("ME_OPENDRT_FORCE_STAGE_COPY");
    if (v == nullptr || v[0] == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

enum class CudaRenderMode {
  HostPreferred,
  InternalOnly
};

enum class MetalRenderMode {
  HostPreferred,
  InternalOnly
};

// Deterministic mode selection (single source of truth):
// - ME_OPENDRT_RENDER_MODE=HOST|AUTO -> host preferred
// - ME_OPENDRT_RENDER_MODE=INTERNAL  -> internal only
// Legacy env vars remain as compatibility fallback.
// Important coupling:
// describe() advertises host-CUDA capability from this selector and render() routes from it.
// Keeping both on the same selector avoids host UI/runtime mismatch.
CudaRenderMode selectedCudaRenderMode() {
  static const CudaRenderMode mode = []() {
    const char* modeVar = std::getenv("ME_OPENDRT_RENDER_MODE");
    if (modeVar && modeVar[0] != '\0') {
      std::string m(modeVar);
      for (char& c : m) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      if (m == "INTERNAL") return CudaRenderMode::InternalOnly;
      if (m == "HOST" || m == "AUTO") return CudaRenderMode::HostPreferred;
    }

    const char* forceInternal = std::getenv("ME_OPENDRT_FORCE_INTERNAL_PATH");
    if (forceInternal && forceInternal[0] != '\0' && !(forceInternal[0] == '0' && forceInternal[1] == '\0')) {
      return CudaRenderMode::InternalOnly;
    }
    const char* hostEnable = std::getenv("ME_OPENDRT_ENABLE_OFX_HOST_CUDA");
    if (hostEnable && hostEnable[0] != '\0' && !(hostEnable[0] == '0' && hostEnable[1] == '\0')) {
      return CudaRenderMode::HostPreferred;
    }

    // Default policy: prefer host-CUDA because it avoids plugin-side staging/copy overhead.
    return CudaRenderMode::HostPreferred;
  }();
  return mode;
}

// Deterministic Metal mode selector:
// - ME_OPENDRT_METAL_RENDER_MODE=HOST|AUTO -> host preferred
// - ME_OPENDRT_METAL_RENDER_MODE=INTERNAL  -> internal-only path
// Default policy mirrors CUDA: prefer host-Metal for lower staging overhead when available.
MetalRenderMode selectedMetalRenderMode() {
  static const MetalRenderMode mode = []() {
    const char* modeVar = std::getenv("ME_OPENDRT_METAL_RENDER_MODE");
    if (modeVar && modeVar[0] != '\0') {
      std::string m(modeVar);
      for (char& c : m) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      if (m == "INTERNAL") return MetalRenderMode::InternalOnly;
      if (m == "HOST" || m == "AUTO") return MetalRenderMode::HostPreferred;
    }
    return MetalRenderMode::HostPreferred;
  }();
  return mode;
}

bool debugLogEnabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("ME_OPENDRT_DEBUG_LOG");
    if (v == nullptr || v[0] == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

// Perf logging writes to stderr and platform-local file locations to help compare host/internal paths.
// Logging is opt-in and non-fatal: any filesystem failures are ignored.
void perfLog(const char* stage, const std::chrono::steady_clock::time_point& start) {
  if (!perfLogEnabled()) return;
  const auto now = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(now - start).count();
  std::fprintf(stderr, "[ME_OpenDRT][PERF] %s: %.3f ms\n", stage, ms);
#if defined(_WIN32)
  static bool pathInit = false;
  static std::filesystem::path logPath;
  if (!pathInit) {
    pathInit = true;
    const char* base = std::getenv("LOCALAPPDATA");
    if (base && *base) {
      logPath = std::filesystem::path(base) / "ME_OpenDRT" / "perf.log";
      std::error_code ec;
      std::filesystem::create_directories(logPath.parent_path(), ec);
    }
  }
  if (!logPath.empty()) {
    std::ofstream ofs(logPath, std::ios::app);
    if (ofs.is_open()) {
      ofs << "[ME_OpenDRT][PERF] " << stage << ": " << ms << " ms\n";
    }
  }
#elif defined(__linux__)
  static bool pathInit = false;
  static std::string logPath;
  if (!pathInit) {
    pathInit = true;
    const char* home = std::getenv("HOME");
    if (home && *home) {
      const std::string cacheDir = std::string(home) + "/.cache";
      const std::string pluginDir = cacheDir + "/ME_OpenDRT";
      (void)::mkdir(cacheDir.c_str(), 0755);
      (void)::mkdir(pluginDir.c_str(), 0755);
      logPath = pluginDir + "/perf.log";
    } else {
      logPath = "/tmp/ME_OpenDRT_perf.log";
    }
  }
  if (!logPath.empty()) {
    FILE* f = std::fopen(logPath.c_str(), "a");
    if (f != nullptr) {
      std::fprintf(f, "[ME_OpenDRT][PERF] %s: %.3f ms\n", stage, ms);
      std::fclose(f);
    }
  }
#endif
}

constexpr int kBuiltInLookPresetCount = static_cast<int>(kLookPresetNames.size());
constexpr int kBuiltInTonescalePresetCount = static_cast<int>(kTonescalePresetNames.size());

// User preset records persisted to presets_v2.json.
// These are host-side settings only and never used in the render kernel hot path.
struct UserLookPreset {
  std::string id;
  std::string name;
  std::string createdAtUtc;
  std::string updatedAtUtc;
  LookPresetValues values{};
};

struct UserTonescalePreset {
  std::string id;
  std::string name;
  std::string createdAtUtc;
  std::string updatedAtUtc;
  TonescalePresetValues values{};
};

struct UserPresetStore {
  bool loaded = false;
  std::vector<UserLookPreset> lookPresets;
  std::vector<UserTonescalePreset> tonescalePresets;
};

// Global in-memory preset cache.
// Access is synchronized by userPresetMutex() for all load/save/update paths.
UserPresetStore& userPresetStore() {
  static UserPresetStore store;
  return store;
}

std::mutex& userPresetMutex() {
  static std::mutex m;
  return m;
}

void ensureUserPresetStoreLoadedLocked();

std::string toLowerCopy(const std::string& s) {
  std::string out = s;
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string normalizePresetNameKey(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  bool inSpace = false;
  for (char c : s) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isspace(uc)) {
      inSpace = true;
      continue;
    }
    if (inSpace && !out.empty()) out.push_back(' ');
    inSpace = false;
    out.push_back(static_cast<char>(std::tolower(uc)));
  }
  while (!out.empty() && out.front() == ' ') out.erase(out.begin());
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

std::string sanitizePresetName(const std::string& s, const char* fallback) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\n' || c == '\r' || c == '\t') continue;
    out.push_back(c);
  }
  while (!out.empty() && out.front() == ' ') out.erase(out.begin());
  while (!out.empty() && out.back() == ' ') out.pop_back();
  if (out.empty()) out = fallback;
  if (out.size() > 96) out.resize(96);
  return out;
}

std::string nowUtcIso8601() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32] = {0};
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return std::string(buf);
}

std::string makePresetId(const std::string& prefix) {
  static unsigned long counter = 1;
  std::ostringstream os;
  os << prefix << '_' << std::time(nullptr) << '_' << counter++;
  return os.str();
}

std::string jsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 16);
  for (char c : in) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

std::string jsonUnescape(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    char c = in[i];
    if (c == '\\' && i + 1 < in.size()) {
      char n = in[++i];
      if (n == 'n') out.push_back('\n');
      else if (n == 'r') out.push_back('\r');
      else if (n == 't') out.push_back('\t');
      else out.push_back(n);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

// ===== Preset File Paths: per-platform storage locations =====
// Resolve user-level preset location.
// Keep path logic centralized so save/import/refresh always resolve consistently.
#if defined(__linux__)
std::string userPresetDirPath() {
  const char* home = std::getenv("HOME");
  if (home && *home) return std::string(home) + "/.config/ME_OpenDRT";
  return ".";
}

std::string userPresetFilePathV2() {
  return userPresetDirPath() + "/presets_v2.json";
}

std::string userPresetFilePathV1Legacy() {
  return userPresetDirPath() + "/user_presets_v1.txt";
}

bool fileExists(const std::string& path) {
  struct stat st {};
  return !path.empty() && ::stat(path.c_str(), &st) == 0;
}

std::string parentPath(const std::string& path) {
  const size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) return ".";
  if (pos == 0) return "/";
  return path.substr(0, pos);
}

bool ensureDirectoryExists(const std::string& dir) {
  if (dir.empty()) return false;
  std::string normalized = dir;
  for (char& c : normalized) {
    if (c == '\\') c = '/';
  }
  std::string current;
  if (!normalized.empty() && normalized[0] == '/') current = "/";
  size_t i = (current == "/") ? 1 : 0;
  while (i < normalized.size()) {
    const size_t slash = normalized.find('/', i);
    const std::string part = normalized.substr(i, slash == std::string::npos ? std::string::npos : slash - i);
    if (!part.empty()) {
      if (!current.empty() && current.back() != '/') current.push_back('/');
      current += part;
      if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
      }
    }
    if (slash == std::string::npos) break;
    i = slash + 1;
  }
  return true;
}
#else
std::filesystem::path userPresetDirPath() {
#ifdef _WIN32
  const char* base = std::getenv("APPDATA");
  if (!base || !*base) base = std::getenv("LOCALAPPDATA");
  if (base && *base) return std::filesystem::path(base) / "ME_OpenDRT";
#elif defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (home && *home) return std::filesystem::path(home) / "Library" / "Application Support" / "ME_OpenDRT";
#else
  const char* home = std::getenv("HOME");
  if (home && *home) return std::filesystem::path(home) / ".config" / "ME_OpenDRT";
#endif
  return std::filesystem::path(".");
}

std::filesystem::path userPresetFilePathV2() {
  return userPresetDirPath() / "presets_v2.json";
}

std::filesystem::path userPresetFilePathV1Legacy() {
  return userPresetDirPath() / "user_presets_v1.txt";
}
#endif

enum class DeleteTarget {
  Cancel = 0,
  Look,
  Tonescale
};

// ===== Dialog + Shell Helpers: platform-specific picker/confirm/url utilities =====
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>

std::string pickOpenJsonFilePath() {
  char filePath[MAX_PATH] = {0};
  OPENFILENAMEA ofn{};
  ofn.lStructSize = sizeof(ofn);
  ofn.lpstrFilter = "JSON Files (*.json)\0*.json\0All Files (*.*)\0*.*\0";
  ofn.lpstrFile = filePath;
  ofn.nMaxFile = MAX_PATH;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
  ofn.lpstrDefExt = "json";
  if (GetOpenFileNameA(&ofn) == TRUE) return std::string(filePath);
  return std::string();
}

std::string pickSaveJsonFilePath(const std::string& defaultName) {
  char filePath[MAX_PATH] = {0};
  std::snprintf(filePath, MAX_PATH, "%s", defaultName.c_str());
  OPENFILENAMEA ofn{};
  ofn.lStructSize = sizeof(ofn);
  ofn.lpstrFilter = "JSON Files (*.json)\0*.json\0All Files (*.*)\0*.*\0";
  ofn.lpstrFile = filePath;
  ofn.nMaxFile = MAX_PATH;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
  ofn.lpstrDefExt = "json";
  if (GetSaveFileNameA(&ofn) == TRUE) return std::string(filePath);
  return std::string();
}

bool confirmOverwriteDialog(const std::string& presetName) {
  std::string msg = "Preset '" + presetName + "' already exists. Overwrite?";
  return MessageBoxA(nullptr, msg.c_str(), "ME_OpenDRT", MB_ICONQUESTION | MB_YESNO) == IDYES;
}

void showInfoDialog(const std::string& text) {
  MessageBoxA(nullptr, text.c_str(), "ME_OpenDRT", MB_ICONINFORMATION | MB_OK);
}

bool confirmDeleteDialog(const std::string& presetName) {
  std::string msg = "Delete preset '" + presetName + "'? This cannot be undone.";
  return MessageBoxA(nullptr, msg.c_str(), "ME_OpenDRT", MB_ICONWARNING | MB_YESNO) == IDYES;
}

DeleteTarget choosePresetTargetDialog(const char* actionVerb) {
  std::string msg = "Both selected Look and Tonescale are user presets.\n\nYes = " + std::string(actionVerb) + " Look\nNo = " + std::string(actionVerb) + " Tonescale\nCancel = Cancel";
  const int result = MessageBoxA(
    nullptr,
    msg.c_str(),
    "ME_OpenDRT",
    MB_ICONQUESTION | MB_YESNOCANCEL
  );
  if (result == IDYES) return DeleteTarget::Look;
  if (result == IDNO) return DeleteTarget::Tonescale;
  return DeleteTarget::Cancel;
}

bool openExternalUrl(const std::string& url) {
  const HINSTANCE rc = ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
  return reinterpret_cast<intptr_t>(rc) > 32;
}
#elif defined(__APPLE__)
std::string execAndRead(const std::string& cmd) {
  std::string out;
  FILE* f = popen(cmd.c_str(), "r");
  if (!f) return out;
  char buf[512];
  while (fgets(buf, sizeof(buf), f)) out += buf;
  pclose(f);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

std::string pickOpenJsonFilePath() {
  return execAndRead("osascript -e 'POSIX path of (choose file with prompt \"Import ME_OpenDRT preset\" of type {\"public.json\"})' 2>/dev/null");
}

std::string pickSaveJsonFilePath(const std::string& defaultName) {
  std::string cmd = "osascript -e 'POSIX path of (choose file name with prompt \"Export ME_OpenDRT preset\" default name \"" + defaultName + "\")' 2>/dev/null";
  return execAndRead(cmd);
}

bool confirmOverwriteDialog(const std::string& presetName) {
  std::string cmd = "osascript -e 'button returned of (display dialog \"Preset \\\"" + presetName + "\\\" already exists. Overwrite?\" buttons {\"Cancel\",\"Overwrite\"} default button \"Overwrite\")' 2>/dev/null";
  return execAndRead(cmd) == "Overwrite";
}

void showInfoDialog(const std::string& text) {
  std::string esc = text;
  for (char& c : esc) if (c == '"') c = '\'';
  std::string cmd = "osascript -e 'display dialog \"" + esc + "\" buttons {\"OK\"} default button \"OK\"' 2>/dev/null";
  (void)execAndRead(cmd);
}

bool confirmDeleteDialog(const std::string& presetName) {
  std::string cmd = "osascript -e 'button returned of (display dialog \"Delete preset \\\"" + presetName + "\\\"? This cannot be undone.\" buttons {\"Cancel\",\"Delete\"} default button \"Delete\")' 2>/dev/null";
  return execAndRead(cmd) == "Delete";
}

DeleteTarget choosePresetTargetDialog(const char* actionVerb) {
  std::string action = actionVerb ? actionVerb : "Apply";
  std::string cmd = "osascript -e 'button returned of (display dialog \"Both selected Look and Tonescale are user presets.\" buttons {\"Cancel\",\"" + action + " Tonescale\",\"" + action + " Look\"} default button \"" + action + " Look\")' 2>/dev/null";
  const std::string out = execAndRead(cmd);
  if (out == (action + " Look")) return DeleteTarget::Look;
  if (out == (action + " Tonescale")) return DeleteTarget::Tonescale;
  return DeleteTarget::Cancel;
}

bool openExternalUrl(const std::string& url) {
  if (url.empty()) return false;
  std::string safe = url;
  for (char& c : safe) {
    if (c == '"') c = '\'';
  }
  std::string cmd = "open \"" + safe + "\" >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}
#else
std::string execAndReadLinux(const std::string& cmd) {
  std::string out;
  FILE* f = popen(cmd.c_str(), "r");
  if (!f) return out;
  char buf[512];
  while (fgets(buf, sizeof(buf), f)) out += buf;
  pclose(f);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

bool linuxCommandExists(const char* cmd) {
  if (cmd == nullptr || cmd[0] == '\0') return false;
  std::string probe = "command -v ";
  probe += cmd;
  probe += " >/dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

std::string pickOpenJsonFilePath() {
  if (linuxCommandExists("zenity")) {
    return execAndReadLinux("zenity --file-selection --title=\"Import ME_OpenDRT preset\" --file-filter=\"*.json\" 2>/dev/null");
  }
  if (linuxCommandExists("kdialog")) {
    return execAndReadLinux("kdialog --getopenfilename \"$HOME\" \"*.json|JSON Files\" 2>/dev/null");
  }
  return std::string();
}

std::string pickSaveJsonFilePath(const std::string& defaultName) {
  if (linuxCommandExists("zenity")) {
    std::string safe = defaultName;
    for (char& c : safe) if (c == '"') c = '\'';
    std::string cmd =
        "zenity --file-selection --save --confirm-overwrite --title=\"Export ME_OpenDRT preset\" --filename=\"$HOME/" +
        safe + "\" 2>/dev/null";
    return execAndReadLinux(cmd);
  }
  if (linuxCommandExists("kdialog")) {
    std::string safe = defaultName;
    for (char& c : safe) if (c == '"') c = '\'';
    std::string cmd = "kdialog --getsavefilename \"$HOME/" + safe + "\" \"*.json|JSON Files\" 2>/dev/null";
    return execAndReadLinux(cmd);
  }
  return std::string();
}

bool confirmOverwriteDialog(const std::string& presetName) {
  if (linuxCommandExists("zenity")) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd = "zenity --question --title=\"ME_OpenDRT\" --text=\"Preset '" + safe +
                            "' already exists. Overwrite?\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
  }
  if (linuxCommandExists("kdialog")) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd = "kdialog --warningyesno \"Preset '" + safe + "' already exists. Overwrite?\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
  }
  std::fprintf(stderr, "[ME_OpenDRT] Linux fallback: overwrite confirmation unavailable for preset '%s'.\n", presetName.c_str());
  return false;
}

void showInfoDialog(const std::string& text) {
  if (linuxCommandExists("zenity")) {
    std::string safe = text;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd = "zenity --info --title=\"ME_OpenDRT\" --text=\"" + safe + "\" 2>/dev/null";
    (void)std::system(cmd.c_str());
    return;
  }
  if (linuxCommandExists("kdialog")) {
    std::string safe = text;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd = "kdialog --msgbox \"" + safe + "\" 2>/dev/null";
    (void)std::system(cmd.c_str());
    return;
  }
  std::fprintf(stderr, "[ME_OpenDRT] %s\n", text.c_str());
}

bool confirmDeleteDialog(const std::string& presetName) {
  if (linuxCommandExists("zenity")) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "zenity --question --title=\"ME_OpenDRT\" --text=\"Delete preset '" + safe + "'? This cannot be undone.\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
  }
  if (linuxCommandExists("kdialog")) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "kdialog --warningyesno \"Delete preset '" + safe + "'? This cannot be undone.\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
  }
  std::fprintf(stderr, "[ME_OpenDRT] Linux fallback: delete confirmation unavailable for preset '%s'.\n", presetName.c_str());
  return false;
}

DeleteTarget choosePresetTargetDialog(const char* actionVerb) {
  const std::string action = actionVerb ? actionVerb : "Apply";
  if (linuxCommandExists("zenity")) {
    std::string safe = action;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "zenity --question --title=\"ME_OpenDRT\" "
        "--ok-label=\"" + safe + " Look\" --cancel-label=\"" + safe +
        " Tonescale\" "
        "--text=\"Both selected Look and Tonescale are user presets.\" 2>/dev/null";
    const int rc = std::system(cmd.c_str());
    if (rc == 0) return DeleteTarget::Look;
    if (rc == 256) return DeleteTarget::Tonescale;
    return DeleteTarget::Cancel;
  }
  if (linuxCommandExists("kdialog")) {
    std::string safe = action;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "kdialog --yesnocancel \"Both selected Look and Tonescale are user presets.\\n\\nYes = " + safe +
        " Look\\nNo = " + safe + " Tonescale\\nCancel = Cancel\" 2>/dev/null";
    const int rc = std::system(cmd.c_str());
    if (rc == 0) return DeleteTarget::Look;
    if (rc == 256) return DeleteTarget::Tonescale;
    return DeleteTarget::Cancel;
  }
  std::fprintf(
      stderr,
      "[ME_OpenDRT] Linux fallback: preset target chooser unavailable for action '%s'.\n",
      action.c_str());
  return DeleteTarget::Cancel;
}

bool openExternalUrl(const std::string& url) {
  if (url.empty()) return false;
  if (!linuxCommandExists("xdg-open")) {
    std::fprintf(stderr, "[ME_OpenDRT] Linux fallback: xdg-open not found.\n");
    return false;
  }
  std::string safe = url;
  for (char& c : safe) {
    if (c == '"') c = '\'';
  }
  const std::string cmd = "xdg-open \"" + safe + "\" >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}
#endif

// ===== Preset Payload Codec: compact canonical payload (schema anchor) =====
// Compact payload serialization keeps files small and load fast.
// Field ordering is versioned-by-convention and should remain stable.
bool serializeLookValues(const LookPresetValues& v, std::string& out) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(9);
  os << v.tn_con << ' ' << v.tn_sh << ' ' << v.tn_toe << ' ' << v.tn_off << ' '
     << v.tn_hcon_enable << ' ' << v.tn_hcon << ' ' << v.tn_hcon_pv << ' ' << v.tn_hcon_st << ' '
     << v.tn_lcon_enable << ' ' << v.tn_lcon << ' ' << v.tn_lcon_w << ' '
     << v.cwp << ' ' << v.cwp_lm << ' '
     << v.rs_sa << ' ' << v.rs_rw << ' ' << v.rs_bw << ' '
     << v.pt_enable << ' '
     << v.pt_lml << ' ' << v.pt_lml_r << ' ' << v.pt_lml_g << ' ' << v.pt_lml_b << ' '
     << v.pt_lmh << ' ' << v.pt_lmh_r << ' ' << v.pt_lmh_b << ' '
     << v.ptl_enable << ' ' << v.ptl_c << ' ' << v.ptl_m << ' ' << v.ptl_y << ' '
     << v.ptm_enable << ' ' << v.ptm_low << ' ' << v.ptm_low_rng << ' ' << v.ptm_low_st << ' '
     << v.ptm_high << ' ' << v.ptm_high_rng << ' ' << v.ptm_high_st << ' '
     << v.brl_enable << ' ' << v.brl << ' ' << v.brl_r << ' ' << v.brl_g << ' ' << v.brl_b << ' '
     << v.brl_rng << ' ' << v.brl_st << ' '
     << v.brlp_enable << ' ' << v.brlp << ' ' << v.brlp_r << ' ' << v.brlp_g << ' ' << v.brlp_b << ' '
     << v.hc_enable << ' ' << v.hc_r << ' ' << v.hc_r_rng << ' '
     << v.hs_rgb_enable << ' ' << v.hs_r << ' ' << v.hs_r_rng << ' '
     << v.hs_g << ' ' << v.hs_g_rng << ' ' << v.hs_b << ' ' << v.hs_b_rng << ' '
     << v.hs_cmy_enable << ' ' << v.hs_c << ' ' << v.hs_c_rng << ' ' << v.hs_m << ' ' << v.hs_m_rng << ' '
     << v.hs_y << ' ' << v.hs_y_rng;
  out = os.str();
  return true;
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
       >> v->hs_y >> v->hs_y_rng
  );
}

bool serializeTonescaleValues(const TonescalePresetValues& v, std::string& out) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(9);
  os << v.tn_con << ' ' << v.tn_sh << ' ' << v.tn_toe << ' ' << v.tn_off << ' '
     << v.tn_hcon_enable << ' ' << v.tn_hcon << ' ' << v.tn_hcon_pv << ' ' << v.tn_hcon_st << ' '
     << v.tn_lcon_enable << ' ' << v.tn_lcon << ' ' << v.tn_lcon_w;
  out = os.str();
  return true;
}

bool parseTonescaleValues(const std::string& in, TonescalePresetValues* v) {
  if (!v) return false;
  std::istringstream is(in);
  return static_cast<bool>(
    is >> v->tn_con >> v->tn_sh >> v->tn_toe >> v->tn_off
       >> v->tn_hcon_enable >> v->tn_hcon >> v->tn_hcon_pv >> v->tn_hcon_st
       >> v->tn_lcon_enable >> v->tn_lcon >> v->tn_lcon_w
  );
}

// ===== Interop String Builders: named JSON + Nuke + DCTL representations =====
std::string formatFloatLiteral(double v) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(9);
  os << v;
  return os.str();
}

std::string nukeEscapeToken(const std::string& s) {
  std::string out;
  out.reserve(s.size() * 2);
  for (char c : s) {
    if (c == '\\' || c == ' ') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

void appendJsonInt(std::ostringstream& os, bool& first, const char* k, int v) {
  if (!first) os << ',';
  first = false;
  os << '\"' << k << "\":" << v;
}

void appendJsonFloat(std::ostringstream& os, bool& first, const char* k, double v) {
  if (!first) os << ',';
  first = false;
  os << '\"' << k << "\":" << formatFloatLiteral(v);
}

void appendNukeInt(std::ostringstream& os, const char* k, int v) {
  os << k << ' ' << v << ' ';
}

void appendNukeFloat(std::ostringstream& os, const char* k, double v) {
  os << k << ' ' << formatFloatLiteral(v) << ' ';
}

void appendDctlInt(std::ostringstream& os, bool& first, const char* k, int v) {
  if (!first) os << ", ";
  first = false;
  os << k << " = " << v;
}

void appendDctlFloat(std::ostringstream& os, bool& first, const char* k, double v) {
  if (!first) os << ", ";
  first = false;
  os << k << " = " << formatFloatLiteral(v) << 'f';
}

std::string lookValuesAsNamedJson(const LookPresetValues& v) {
  std::ostringstream os;
  os << '{';
  bool first = true;
  appendJsonFloat(os, first, "tn_con", v.tn_con); appendJsonFloat(os, first, "tn_sh", v.tn_sh);
  appendJsonFloat(os, first, "tn_toe", v.tn_toe); appendJsonFloat(os, first, "tn_off", v.tn_off);
  appendJsonInt(os, first, "tn_hcon_enable", v.tn_hcon_enable); appendJsonFloat(os, first, "tn_hcon", v.tn_hcon);
  appendJsonFloat(os, first, "tn_hcon_pv", v.tn_hcon_pv); appendJsonFloat(os, first, "tn_hcon_st", v.tn_hcon_st);
  appendJsonInt(os, first, "tn_lcon_enable", v.tn_lcon_enable); appendJsonFloat(os, first, "tn_lcon", v.tn_lcon);
  appendJsonFloat(os, first, "tn_lcon_w", v.tn_lcon_w); appendJsonInt(os, first, "cwp", v.cwp);
  appendJsonFloat(os, first, "cwp_lm", v.cwp_lm); appendJsonFloat(os, first, "rs_sa", v.rs_sa);
  appendJsonFloat(os, first, "rs_rw", v.rs_rw); appendJsonFloat(os, first, "rs_bw", v.rs_bw);
  appendJsonInt(os, first, "pt_enable", v.pt_enable); appendJsonFloat(os, first, "pt_lml", v.pt_lml);
  appendJsonFloat(os, first, "pt_lml_r", v.pt_lml_r); appendJsonFloat(os, first, "pt_lml_g", v.pt_lml_g);
  appendJsonFloat(os, first, "pt_lml_b", v.pt_lml_b); appendJsonFloat(os, first, "pt_lmh", v.pt_lmh);
  appendJsonFloat(os, first, "pt_lmh_r", v.pt_lmh_r); appendJsonFloat(os, first, "pt_lmh_b", v.pt_lmh_b);
  appendJsonInt(os, first, "ptl_enable", v.ptl_enable); appendJsonFloat(os, first, "ptl_c", v.ptl_c);
  appendJsonFloat(os, first, "ptl_m", v.ptl_m); appendJsonFloat(os, first, "ptl_y", v.ptl_y);
  appendJsonInt(os, first, "ptm_enable", v.ptm_enable); appendJsonFloat(os, first, "ptm_low", v.ptm_low);
  appendJsonFloat(os, first, "ptm_low_rng", v.ptm_low_rng); appendJsonFloat(os, first, "ptm_low_st", v.ptm_low_st);
  appendJsonFloat(os, first, "ptm_high", v.ptm_high); appendJsonFloat(os, first, "ptm_high_rng", v.ptm_high_rng);
  appendJsonFloat(os, first, "ptm_high_st", v.ptm_high_st); appendJsonInt(os, first, "brl_enable", v.brl_enable);
  appendJsonFloat(os, first, "brl", v.brl); appendJsonFloat(os, first, "brl_r", v.brl_r);
  appendJsonFloat(os, first, "brl_g", v.brl_g); appendJsonFloat(os, first, "brl_b", v.brl_b);
  appendJsonFloat(os, first, "brl_rng", v.brl_rng); appendJsonFloat(os, first, "brl_st", v.brl_st);
  appendJsonInt(os, first, "brlp_enable", v.brlp_enable); appendJsonFloat(os, first, "brlp", v.brlp);
  appendJsonFloat(os, first, "brlp_r", v.brlp_r); appendJsonFloat(os, first, "brlp_g", v.brlp_g);
  appendJsonFloat(os, first, "brlp_b", v.brlp_b); appendJsonInt(os, first, "hc_enable", v.hc_enable);
  appendJsonFloat(os, first, "hc_r", v.hc_r); appendJsonFloat(os, first, "hc_r_rng", v.hc_r_rng);
  appendJsonInt(os, first, "hs_rgb_enable", v.hs_rgb_enable); appendJsonFloat(os, first, "hs_r", v.hs_r);
  appendJsonFloat(os, first, "hs_r_rng", v.hs_r_rng); appendJsonFloat(os, first, "hs_g", v.hs_g);
  appendJsonFloat(os, first, "hs_g_rng", v.hs_g_rng); appendJsonFloat(os, first, "hs_b", v.hs_b);
  appendJsonFloat(os, first, "hs_b_rng", v.hs_b_rng); appendJsonInt(os, first, "hs_cmy_enable", v.hs_cmy_enable);
  appendJsonFloat(os, first, "hs_c", v.hs_c); appendJsonFloat(os, first, "hs_c_rng", v.hs_c_rng);
  appendJsonFloat(os, first, "hs_m", v.hs_m); appendJsonFloat(os, first, "hs_m_rng", v.hs_m_rng);
  appendJsonFloat(os, first, "hs_y", v.hs_y); appendJsonFloat(os, first, "hs_y_rng", v.hs_y_rng);
  os << '}';
  return os.str();
}

std::string tonescaleValuesAsNamedJson(const TonescalePresetValues& v) {
  std::ostringstream os;
  os << '{';
  bool first = true;
  appendJsonFloat(os, first, "tn_con", v.tn_con); appendJsonFloat(os, first, "tn_sh", v.tn_sh);
  appendJsonFloat(os, first, "tn_toe", v.tn_toe); appendJsonFloat(os, first, "tn_off", v.tn_off);
  appendJsonInt(os, first, "tn_hcon_enable", v.tn_hcon_enable); appendJsonFloat(os, first, "tn_hcon", v.tn_hcon);
  appendJsonFloat(os, first, "tn_hcon_pv", v.tn_hcon_pv); appendJsonFloat(os, first, "tn_hcon_st", v.tn_hcon_st);
  appendJsonInt(os, first, "tn_lcon_enable", v.tn_lcon_enable); appendJsonFloat(os, first, "tn_lcon", v.tn_lcon);
  appendJsonFloat(os, first, "tn_lcon_w", v.tn_lcon_w);
  os << '}';
  return os.str();
}

std::string lookValuesAsNukeCmd(const std::string& presetName, const LookPresetValues& v) {
  std::ostringstream os;
  os << "knobs this {";
  appendNukeFloat(os, "tn_con", v.tn_con); appendNukeFloat(os, "tn_sh", v.tn_sh);
  appendNukeFloat(os, "tn_toe", v.tn_toe); appendNukeFloat(os, "tn_off", v.tn_off);
  appendNukeInt(os, "tn_hcon_enable", v.tn_hcon_enable); appendNukeFloat(os, "tn_hcon", v.tn_hcon);
  appendNukeFloat(os, "tn_hcon_pv", v.tn_hcon_pv); appendNukeFloat(os, "tn_hcon_st", v.tn_hcon_st);
  appendNukeInt(os, "tn_lcon_enable", v.tn_lcon_enable); appendNukeFloat(os, "tn_lcon", v.tn_lcon);
  appendNukeFloat(os, "tn_lcon_w", v.tn_lcon_w); appendNukeInt(os, "cwp", v.cwp);
  appendNukeFloat(os, "cwp_lm", v.cwp_lm); appendNukeFloat(os, "rs_sa", v.rs_sa);
  appendNukeFloat(os, "rs_rw", v.rs_rw); appendNukeFloat(os, "rs_bw", v.rs_bw);
  appendNukeInt(os, "pt_enable", v.pt_enable); appendNukeFloat(os, "pt_lml", v.pt_lml);
  appendNukeFloat(os, "pt_lml_r", v.pt_lml_r); appendNukeFloat(os, "pt_lml_g", v.pt_lml_g);
  appendNukeFloat(os, "pt_lml_b", v.pt_lml_b); appendNukeFloat(os, "pt_lmh", v.pt_lmh);
  appendNukeFloat(os, "pt_lmh_r", v.pt_lmh_r); appendNukeFloat(os, "pt_lmh_b", v.pt_lmh_b);
  appendNukeInt(os, "ptl_enable", v.ptl_enable); appendNukeFloat(os, "ptl_c", v.ptl_c);
  appendNukeFloat(os, "ptl_m", v.ptl_m); appendNukeFloat(os, "ptl_y", v.ptl_y);
  appendNukeInt(os, "ptm_enable", v.ptm_enable); appendNukeFloat(os, "ptm_low", v.ptm_low);
  appendNukeFloat(os, "ptm_low_rng", v.ptm_low_rng); appendNukeFloat(os, "ptm_low_st", v.ptm_low_st);
  appendNukeFloat(os, "ptm_high", v.ptm_high); appendNukeFloat(os, "ptm_high_rng", v.ptm_high_rng);
  appendNukeFloat(os, "ptm_high_st", v.ptm_high_st); appendNukeInt(os, "brl_enable", v.brl_enable);
  appendNukeFloat(os, "brl", v.brl); appendNukeFloat(os, "brl_r", v.brl_r);
  appendNukeFloat(os, "brl_g", v.brl_g); appendNukeFloat(os, "brl_b", v.brl_b);
  appendNukeFloat(os, "brl_rng", v.brl_rng); appendNukeFloat(os, "brl_st", v.brl_st);
  appendNukeInt(os, "brlp_enable", v.brlp_enable); appendNukeFloat(os, "brlp", v.brlp);
  appendNukeFloat(os, "brlp_r", v.brlp_r); appendNukeFloat(os, "brlp_g", v.brlp_g);
  appendNukeFloat(os, "brlp_b", v.brlp_b); appendNukeInt(os, "hc_enable", v.hc_enable);
  appendNukeFloat(os, "hc_r", v.hc_r); appendNukeFloat(os, "hc_r_rng", v.hc_r_rng);
  appendNukeInt(os, "hs_rgb_enable", v.hs_rgb_enable); appendNukeFloat(os, "hs_r", v.hs_r);
  appendNukeFloat(os, "hs_r_rng", v.hs_r_rng); appendNukeFloat(os, "hs_g", v.hs_g);
  appendNukeFloat(os, "hs_g_rng", v.hs_g_rng); appendNukeFloat(os, "hs_b", v.hs_b);
  appendNukeFloat(os, "hs_b_rng", v.hs_b_rng); appendNukeInt(os, "hs_cmy_enable", v.hs_cmy_enable);
  appendNukeFloat(os, "hs_c", v.hs_c); appendNukeFloat(os, "hs_c_rng", v.hs_c_rng);
  appendNukeFloat(os, "hs_m", v.hs_m); appendNukeFloat(os, "hs_m_rng", v.hs_m_rng);
  appendNukeFloat(os, "hs_y", v.hs_y); appendNukeFloat(os, "hs_y_rng", v.hs_y_rng);
  os << "look_name " << nukeEscapeToken(presetName) << "}";
  return os.str();
}

std::string tonescaleValuesAsNukeCmd(const TonescalePresetValues& v) {
  std::ostringstream os;
  os << "knobs this {";
  appendNukeFloat(os, "tn_con", v.tn_con); appendNukeFloat(os, "tn_sh", v.tn_sh);
  appendNukeFloat(os, "tn_toe", v.tn_toe); appendNukeFloat(os, "tn_off", v.tn_off);
  appendNukeInt(os, "tn_hcon_enable", v.tn_hcon_enable); appendNukeFloat(os, "tn_hcon", v.tn_hcon);
  appendNukeFloat(os, "tn_hcon_pv", v.tn_hcon_pv); appendNukeFloat(os, "tn_hcon_st", v.tn_hcon_st);
  appendNukeInt(os, "tn_lcon_enable", v.tn_lcon_enable); appendNukeFloat(os, "tn_lcon", v.tn_lcon);
  appendNukeFloat(os, "tn_lcon_w", v.tn_lcon_w);
  os << "}";
  return os.str();
}

std::string lookValuesAsDctl(const LookPresetValues& v) {
  std::ostringstream os;
  bool first = true;
  appendDctlFloat(os, first, "tn_con", v.tn_con); appendDctlFloat(os, first, "tn_sh", v.tn_sh);
  appendDctlFloat(os, first, "tn_toe", v.tn_toe); appendDctlFloat(os, first, "tn_off", v.tn_off);
  appendDctlInt(os, first, "tn_hcon_enable", v.tn_hcon_enable); appendDctlFloat(os, first, "tn_hcon", v.tn_hcon);
  appendDctlFloat(os, first, "tn_hcon_pv", v.tn_hcon_pv); appendDctlFloat(os, first, "tn_hcon_st", v.tn_hcon_st);
  appendDctlInt(os, first, "tn_lcon_enable", v.tn_lcon_enable); appendDctlFloat(os, first, "tn_lcon", v.tn_lcon);
  appendDctlFloat(os, first, "tn_lcon_w", v.tn_lcon_w); appendDctlInt(os, first, "cwp", v.cwp);
  appendDctlFloat(os, first, "cwp_lm", v.cwp_lm); appendDctlFloat(os, first, "rs_sa", v.rs_sa);
  appendDctlFloat(os, first, "rs_rw", v.rs_rw); appendDctlFloat(os, first, "rs_bw", v.rs_bw);
  appendDctlInt(os, first, "pt_enable", v.pt_enable); appendDctlFloat(os, first, "pt_lml", v.pt_lml);
  appendDctlFloat(os, first, "pt_lml_r", v.pt_lml_r); appendDctlFloat(os, first, "pt_lml_g", v.pt_lml_g);
  appendDctlFloat(os, first, "pt_lml_b", v.pt_lml_b); appendDctlFloat(os, first, "pt_lmh", v.pt_lmh);
  appendDctlFloat(os, first, "pt_lmh_r", v.pt_lmh_r); appendDctlFloat(os, first, "pt_lmh_b", v.pt_lmh_b);
  appendDctlInt(os, first, "ptl_enable", v.ptl_enable); appendDctlFloat(os, first, "ptl_c", v.ptl_c);
  appendDctlFloat(os, first, "ptl_m", v.ptl_m); appendDctlFloat(os, first, "ptl_y", v.ptl_y);
  appendDctlInt(os, first, "ptm_enable", v.ptm_enable); appendDctlFloat(os, first, "ptm_low", v.ptm_low);
  appendDctlFloat(os, first, "ptm_low_rng", v.ptm_low_rng); appendDctlFloat(os, first, "ptm_low_st", v.ptm_low_st);
  appendDctlFloat(os, first, "ptm_high", v.ptm_high); appendDctlFloat(os, first, "ptm_high_rng", v.ptm_high_rng);
  appendDctlFloat(os, first, "ptm_high_st", v.ptm_high_st); appendDctlInt(os, first, "brl_enable", v.brl_enable);
  appendDctlFloat(os, first, "brl", v.brl); appendDctlFloat(os, first, "brl_r", v.brl_r);
  appendDctlFloat(os, first, "brl_g", v.brl_g); appendDctlFloat(os, first, "brl_b", v.brl_b);
  appendDctlFloat(os, first, "brl_rng", v.brl_rng); appendDctlFloat(os, first, "brl_st", v.brl_st);
  appendDctlInt(os, first, "brlp_enable", v.brlp_enable); appendDctlFloat(os, first, "brlp", v.brlp);
  appendDctlFloat(os, first, "brlp_r", v.brlp_r); appendDctlFloat(os, first, "brlp_g", v.brlp_g);
  appendDctlFloat(os, first, "brlp_b", v.brlp_b); appendDctlInt(os, first, "hc_enable", v.hc_enable);
  appendDctlFloat(os, first, "hc_r", v.hc_r); appendDctlFloat(os, first, "hc_r_rng", v.hc_r_rng);
  appendDctlInt(os, first, "hs_rgb_enable", v.hs_rgb_enable); appendDctlFloat(os, first, "hs_r", v.hs_r);
  appendDctlFloat(os, first, "hs_r_rng", v.hs_r_rng); appendDctlFloat(os, first, "hs_g", v.hs_g);
  appendDctlFloat(os, first, "hs_g_rng", v.hs_g_rng); appendDctlFloat(os, first, "hs_b", v.hs_b);
  appendDctlFloat(os, first, "hs_b_rng", v.hs_b_rng); appendDctlInt(os, first, "hs_cmy_enable", v.hs_cmy_enable);
  appendDctlFloat(os, first, "hs_c", v.hs_c); appendDctlFloat(os, first, "hs_c_rng", v.hs_c_rng);
  appendDctlFloat(os, first, "hs_m", v.hs_m); appendDctlFloat(os, first, "hs_m_rng", v.hs_m_rng);
  appendDctlFloat(os, first, "hs_y", v.hs_y); appendDctlFloat(os, first, "hs_y_rng", v.hs_y_rng);
  os << ';';
  return os.str();
}

std::string tonescaleValuesAsDctl(const TonescalePresetValues& v) {
  std::ostringstream os;
  bool first = true;
  appendDctlFloat(os, first, "tn_con", v.tn_con); appendDctlFloat(os, first, "tn_sh", v.tn_sh);
  appendDctlFloat(os, first, "tn_toe", v.tn_toe); appendDctlFloat(os, first, "tn_off", v.tn_off);
  appendDctlInt(os, first, "tn_hcon_enable", v.tn_hcon_enable); appendDctlFloat(os, first, "tn_hcon", v.tn_hcon);
  appendDctlFloat(os, first, "tn_hcon_pv", v.tn_hcon_pv); appendDctlFloat(os, first, "tn_hcon_st", v.tn_hcon_st);
  appendDctlInt(os, first, "tn_lcon_enable", v.tn_lcon_enable); appendDctlFloat(os, first, "tn_lcon", v.tn_lcon);
  appendDctlFloat(os, first, "tn_lcon_w", v.tn_lcon_w);
  os << ';';
  return os.str();
}

// ===== JSON Object Utilities: lightweight field extraction/pretty printing =====
std::string jsonField(const std::string& line, const std::string& key) {
  const std::string token = "\"" + key + "\":\"";
  const size_t p = line.find(token);
  if (p == std::string::npos) return std::string();
  size_t i = p + token.size();
  std::string out;
  bool esc = false;
  for (; i < line.size(); ++i) {
    char c = line[i];
    if (esc) {
      out.push_back('\\');
      out.push_back(c);
      esc = false;
      continue;
    }
    if (c == '\\') { esc = true; continue; }
    if (c == '"') break;
    out.push_back(c);
  }
  return jsonUnescape(out);
}

std::string jsonObjectField(const std::string& text, const std::string& key) {
  const std::string token = "\"" + key + "\":";
  const size_t p = text.find(token);
  if (p == std::string::npos) return std::string();
  size_t i = p + token.size();
  while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
  if (i >= text.size() || text[i] != '{') return std::string();
  const size_t start = i;
  int depth = 0;
  bool inString = false;
  bool esc = false;
  for (; i < text.size(); ++i) {
    const char c = text[i];
    if (inString) {
      if (esc) {
        esc = false;
      } else if (c == '\\') {
        esc = true;
      } else if (c == '"') {
        inString = false;
      }
      continue;
    }
    if (c == '"') {
      inString = true;
      continue;
    }
    if (c == '{') {
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0) return text.substr(start, i - start + 1);
    }
  }
  return std::string();
}

std::string prettyJsonObject(const std::string& compact, const std::string& indent) {
  if (compact.empty()) return "{}";
  std::ostringstream out;
  int depth = 0;
  bool inString = false;
  bool esc = false;
  for (size_t i = 0; i < compact.size(); ++i) {
    const char c = compact[i];
    if (inString) {
      out << c;
      if (esc) {
        esc = false;
      } else if (c == '\\') {
        esc = true;
      } else if (c == '"') {
        inString = false;
      }
      continue;
    }
    if (c == '"') {
      inString = true;
      out << c;
      continue;
    }
    if (c == '{') {
      out << c;
      ++depth;
      out << '\n' << indent << std::string(depth * 2, ' ');
      continue;
    }
    if (c == '}') {
      --depth;
      out << '\n' << indent << std::string(depth * 2, ' ') << c;
      continue;
    }
    if (c == ',') {
      out << c << '\n' << indent << std::string(depth * 2, ' ');
      continue;
    }
    out << c;
  }
  return out.str();
}

bool extractJsonObjectFromStream(std::istream& is, const std::string& firstLine, std::string* outObj) {
  if (!outObj) return false;
  std::string obj = firstLine;
  int depth = 0;
  bool inString = false;
  bool esc = false;
  auto scan = [&](const std::string& s) {
    for (char c : s) {
      if (inString) {
        if (esc) {
          esc = false;
        } else if (c == '\\') {
          esc = true;
        } else if (c == '"') {
          inString = false;
        }
        continue;
      }
      if (c == '"') {
        inString = true;
      } else if (c == '{') {
        ++depth;
      } else if (c == '}') {
        --depth;
      }
    }
  };
  scan(firstLine);
  while (depth > 0 && is.good()) {
    std::string next;
    if (!std::getline(is, next)) break;
    obj.append("\n");
    obj.append(next);
    scan(next);
  }
  if (depth != 0) return false;
  *outObj = obj;
  return true;
}

bool jsonNumberField(const std::string& obj, const char* key, double* out) {
  if (!out || !key) return false;
  const std::string token = std::string("\"") + key + "\":";
  const size_t p = obj.find(token);
  if (p == std::string::npos) return false;
  size_t i = p + token.size();
  while (i < obj.size() && std::isspace(static_cast<unsigned char>(obj[i]))) ++i;
  if (i >= obj.size()) return false;
  char* endp = nullptr;
  const double v = std::strtod(obj.c_str() + i, &endp);
  if (endp == obj.c_str() + i) return false;
  *out = v;
  return true;
}

template <typename T>
bool jsonNumberFieldAs(const std::string& obj, const char* key, T* out) {
  if (!out) return false;
  double v = 0.0;
  if (!jsonNumberField(obj, key, &v)) return false;
  *out = static_cast<T>(v);
  return true;
}

bool parseLookValuesFromNamedJson(const std::string& obj, LookPresetValues* v) {
  if (!v || obj.empty()) return false;
  auto reqD = [&](const char* k, auto* d) -> bool { return jsonNumberFieldAs(obj, k, d); };
  auto reqI = [&](const char* k, int* d) -> bool {
    double x = 0.0;
    if (!jsonNumberField(obj, k, &x)) return false;
    *d = static_cast<int>(std::llround(x));
    return true;
  };
  return reqD("tn_con", &v->tn_con) && reqD("tn_sh", &v->tn_sh) && reqD("tn_toe", &v->tn_toe) && reqD("tn_off", &v->tn_off) &&
         reqI("tn_hcon_enable", &v->tn_hcon_enable) && reqD("tn_hcon", &v->tn_hcon) && reqD("tn_hcon_pv", &v->tn_hcon_pv) && reqD("tn_hcon_st", &v->tn_hcon_st) &&
         reqI("tn_lcon_enable", &v->tn_lcon_enable) && reqD("tn_lcon", &v->tn_lcon) && reqD("tn_lcon_w", &v->tn_lcon_w) &&
         reqI("cwp", &v->cwp) && reqD("cwp_lm", &v->cwp_lm) &&
         reqD("rs_sa", &v->rs_sa) && reqD("rs_rw", &v->rs_rw) && reqD("rs_bw", &v->rs_bw) &&
         reqI("pt_enable", &v->pt_enable) &&
         reqD("pt_lml", &v->pt_lml) && reqD("pt_lml_r", &v->pt_lml_r) && reqD("pt_lml_g", &v->pt_lml_g) && reqD("pt_lml_b", &v->pt_lml_b) &&
         reqD("pt_lmh", &v->pt_lmh) && reqD("pt_lmh_r", &v->pt_lmh_r) && reqD("pt_lmh_b", &v->pt_lmh_b) &&
         reqI("ptl_enable", &v->ptl_enable) && reqD("ptl_c", &v->ptl_c) && reqD("ptl_m", &v->ptl_m) && reqD("ptl_y", &v->ptl_y) &&
         reqI("ptm_enable", &v->ptm_enable) && reqD("ptm_low", &v->ptm_low) && reqD("ptm_low_rng", &v->ptm_low_rng) && reqD("ptm_low_st", &v->ptm_low_st) &&
         reqD("ptm_high", &v->ptm_high) && reqD("ptm_high_rng", &v->ptm_high_rng) && reqD("ptm_high_st", &v->ptm_high_st) &&
         reqI("brl_enable", &v->brl_enable) && reqD("brl", &v->brl) && reqD("brl_r", &v->brl_r) && reqD("brl_g", &v->brl_g) && reqD("brl_b", &v->brl_b) &&
         reqD("brl_rng", &v->brl_rng) && reqD("brl_st", &v->brl_st) &&
         reqI("brlp_enable", &v->brlp_enable) && reqD("brlp", &v->brlp) && reqD("brlp_r", &v->brlp_r) && reqD("brlp_g", &v->brlp_g) && reqD("brlp_b", &v->brlp_b) &&
         reqI("hc_enable", &v->hc_enable) && reqD("hc_r", &v->hc_r) && reqD("hc_r_rng", &v->hc_r_rng) &&
         reqI("hs_rgb_enable", &v->hs_rgb_enable) && reqD("hs_r", &v->hs_r) && reqD("hs_r_rng", &v->hs_r_rng) &&
         reqD("hs_g", &v->hs_g) && reqD("hs_g_rng", &v->hs_g_rng) && reqD("hs_b", &v->hs_b) && reqD("hs_b_rng", &v->hs_b_rng) &&
         reqI("hs_cmy_enable", &v->hs_cmy_enable) && reqD("hs_c", &v->hs_c) && reqD("hs_c_rng", &v->hs_c_rng) &&
         reqD("hs_m", &v->hs_m) && reqD("hs_m_rng", &v->hs_m_rng) && reqD("hs_y", &v->hs_y) && reqD("hs_y_rng", &v->hs_y_rng);
}

bool parseTonescaleValuesFromNamedJson(const std::string& obj, TonescalePresetValues* v) {
  if (!v || obj.empty()) return false;
  auto reqD = [&](const char* k, auto* d) -> bool { return jsonNumberFieldAs(obj, k, d); };
  auto reqI = [&](const char* k, int* d) -> bool {
    double x = 0.0;
    if (!jsonNumberField(obj, k, &x)) return false;
    *d = static_cast<int>(std::llround(x));
    return true;
  };
  return reqD("tn_con", &v->tn_con) && reqD("tn_sh", &v->tn_sh) && reqD("tn_toe", &v->tn_toe) && reqD("tn_off", &v->tn_off) &&
         reqI("tn_hcon_enable", &v->tn_hcon_enable) && reqD("tn_hcon", &v->tn_hcon) && reqD("tn_hcon_pv", &v->tn_hcon_pv) &&
         reqD("tn_hcon_st", &v->tn_hcon_st) && reqI("tn_lcon_enable", &v->tn_lcon_enable) && reqD("tn_lcon", &v->tn_lcon) &&
         reqD("tn_lcon_w", &v->tn_lcon_w);
}

// Persist the in-memory preset store to disk using schema v3.
// payload stays canonical for backward compatibility; named/nuke/dctl are derived views.
// ===== Preset Store: Serialize in-memory user presets to JSON on disk =====
void saveUserPresetStoreLocked() {
  const auto path = userPresetFilePathV2();
#if defined(__linux__)
  (void)ensureDirectoryExists(parentPath(path));
#else
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
#endif
  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if (!os.is_open()) return;

  UserPresetStore& s = userPresetStore();
  os << "{\n";
  os << "  \"schemaVersion\":3,\n";
  os << "  \"updatedAtUtc\":\"" << jsonEscape(nowUtcIso8601()) << "\",\n";
  os << "  \"lookPresets\":[\n";
  for (size_t i = 0; i < s.lookPresets.size(); ++i) {
    std::string payload;
    serializeLookValues(s.lookPresets[i].values, payload);
    const std::string named = prettyJsonObject(lookValuesAsNamedJson(s.lookPresets[i].values), "      ");
    const std::string nukeCmd = lookValuesAsNukeCmd(s.lookPresets[i].name, s.lookPresets[i].values);
    const std::string nukeMenu = std::string("Look Presets/") + s.lookPresets[i].name;
    const std::string dctl = lookValuesAsDctl(s.lookPresets[i].values);
    os << "    {\n";
    os << "      \"id\":\"" << jsonEscape(s.lookPresets[i].id) << "\",\n";
    os << "      \"name\":\"" << jsonEscape(s.lookPresets[i].name) << "\",\n";
    os << "      \"createdAtUtc\":\"" << jsonEscape(s.lookPresets[i].createdAtUtc) << "\",\n";
    os << "      \"updatedAtUtc\":\"" << jsonEscape(s.lookPresets[i].updatedAtUtc) << "\",\n";
    os << "      \"payload\":\"" << jsonEscape(payload) << "\",\n";
    os << "      \"namedValues\":" << named << ",\n";
    os << "      \"nuke\":{\n";
    os << "        \"menuEntry\":\"" << jsonEscape(nukeMenu) << "\",\n";
    os << "        \"presetCmd\":\"" << jsonEscape(nukeCmd) << "\"\n";
    os << "      },\n";
    os << "      \"dctl\":{\n";
    os << "        \"preset\":\"" << jsonEscape(dctl) << "\"\n";
    os << "      }\n";
    os << "    }";
    os << (i + 1 < s.lookPresets.size() ? ",\n" : "\n");
  }
  os << "  ],\n";
  os << "  \"tonescalePresets\":[\n";
  for (size_t i = 0; i < s.tonescalePresets.size(); ++i) {
    std::string payload;
    serializeTonescaleValues(s.tonescalePresets[i].values, payload);
    const std::string named = prettyJsonObject(tonescaleValuesAsNamedJson(s.tonescalePresets[i].values), "      ");
    const std::string nukeCmd = tonescaleValuesAsNukeCmd(s.tonescalePresets[i].values);
    const std::string nukeMenu = std::string("Tonescale Presets/") + s.tonescalePresets[i].name;
    const std::string dctl = tonescaleValuesAsDctl(s.tonescalePresets[i].values);
    os << "    {\n";
    os << "      \"id\":\"" << jsonEscape(s.tonescalePresets[i].id) << "\",\n";
    os << "      \"name\":\"" << jsonEscape(s.tonescalePresets[i].name) << "\",\n";
    os << "      \"createdAtUtc\":\"" << jsonEscape(s.tonescalePresets[i].createdAtUtc) << "\",\n";
    os << "      \"updatedAtUtc\":\"" << jsonEscape(s.tonescalePresets[i].updatedAtUtc) << "\",\n";
    os << "      \"payload\":\"" << jsonEscape(payload) << "\",\n";
    os << "      \"namedValues\":" << named << ",\n";
    os << "      \"nuke\":{\n";
    os << "        \"menuEntry\":\"" << jsonEscape(nukeMenu) << "\",\n";
    os << "        \"presetCmd\":\"" << jsonEscape(nukeCmd) << "\"\n";
    os << "      },\n";
    os << "      \"dctl\":{\n";
    os << "        \"preset\":\"" << jsonEscape(dctl) << "\"\n";
    os << "      }\n";
    os << "    }";
    os << (i + 1 < s.tonescalePresets.size() ? ",\n" : "\n");
  }
  os << "  ]\n";
  os << "}\n";
}

// One-time compatibility migration from legacy v1 format when v2 does not exist.
void migrateLegacyV1IfNeededLocked() {
  const auto v2 = userPresetFilePathV2();
#if defined(__linux__)
  if (fileExists(v2)) return;
#else
  if (std::filesystem::exists(v2)) return;
#endif
  std::ifstream is(userPresetFilePathV1Legacy(), std::ios::binary);
  if (!is.is_open()) return;

  std::string header;
  std::getline(is, header);
  if (header != "ME_OPENDRT_USER_PRESETS_V1") return;

  UserPresetStore& s = userPresetStore();
  std::unordered_map<std::string, bool> seenLookNames;
  std::unordered_map<std::string, bool> seenToneNames;
  for (const char* n : kLookPresetNames) seenLookNames[normalizePresetNameKey(n)] = true;
  for (const char* n : kTonescalePresetNames) seenToneNames[normalizePresetNameKey(n)] = true;
  std::string line;
  while (std::getline(is, line)) {
    if (line.empty()) continue;
    const size_t p1 = line.find('\t');
    if (p1 == std::string::npos) continue;
    const size_t p2 = line.find('\t', p1 + 1);
    if (p2 == std::string::npos) continue;
    const size_t p3 = line.find('\t', p2 + 1);
    if (p3 == std::string::npos) continue;
    const std::string kind = line.substr(0, p1);
    const std::string name = sanitizePresetName(line.substr(p2 + 1, p3 - p2 - 1), "User Preset");
    const std::string values = line.substr(p3 + 1);
    const std::string now = nowUtcIso8601();
    if (kind == "LOOK") {
      const std::string key = normalizePresetNameKey(name);
      if (seenLookNames.find(key) != seenLookNames.end()) continue;
      LookPresetValues parsed{};
      if (parseLookValues(values, &parsed)) {
        UserLookPreset p{};
        p.id = makePresetId("look"); p.name = name; p.createdAtUtc = now; p.updatedAtUtc = now; p.values = parsed;
        s.lookPresets.push_back(p);
        seenLookNames[key] = true;
      }
    } else if (kind == "TONE") {
      const std::string key = normalizePresetNameKey(name);
      if (seenToneNames.find(key) != seenToneNames.end()) continue;
      TonescalePresetValues parsed{};
      if (parseTonescaleValues(values, &parsed)) {
        UserTonescalePreset p{};
        p.id = makePresetId("tone"); p.name = name; p.createdAtUtc = now; p.updatedAtUtc = now; p.values = parsed;
        s.tonescalePresets.push_back(p);
        seenToneNames[key] = true;
      }
    }
  }
  saveUserPresetStoreLocked();
}

// Lazy-load preset storage into memory.
// Reader accepts v2/v3 style records and prefers payload parsing, then namedValues fallback.
// Callers must hold userPresetMutex() before calling this helper.
// ===== Preset Store: Load + migrate user presets into memory cache =====
void ensureUserPresetStoreLoadedLocked() {
  UserPresetStore& s = userPresetStore();
  if (s.loaded) return;
  s = UserPresetStore{};
  s.loaded = true;

  migrateLegacyV1IfNeededLocked();

  std::ifstream is(userPresetFilePathV2(), std::ios::binary);
  if (!is.is_open()) return;

  enum class Section { None, Look, Tone };
  Section sec = Section::None;
  std::unordered_map<std::string, bool> seenLookNames;
  std::unordered_map<std::string, bool> seenToneNames;
  for (const char* n : kLookPresetNames) seenLookNames[normalizePresetNameKey(n)] = true;
  for (const char* n : kTonescalePresetNames) seenToneNames[normalizePresetNameKey(n)] = true;
  std::string line;
  while (std::getline(is, line)) {
    if (line.find("\"lookPresets\"") != std::string::npos) { sec = Section::Look; continue; }
    if (line.find("\"tonescalePresets\"") != std::string::npos) { sec = Section::Tone; continue; }
    if (sec == Section::None) continue;
    if (line.find(']') != std::string::npos) { sec = Section::None; continue; }
    if (line.find('{') == std::string::npos) continue;

    std::string obj;
    if (!extractJsonObjectFromStream(is, line, &obj)) continue;
    if (obj.find("\"id\"") == std::string::npos) continue;

    const std::string id = jsonField(obj, "id");
    const std::string name = sanitizePresetName(jsonField(obj, "name"), "User Preset");
    const std::string created = jsonField(obj, "createdAtUtc");
    const std::string updated = jsonField(obj, "updatedAtUtc");
    std::string payload = jsonField(obj, "payload");
    const std::string namedValues = jsonObjectField(obj, "namedValues");
    if (id.empty()) continue;

    if (sec == Section::Look) {
      const std::string key = normalizePresetNameKey(name);
      if (seenLookNames.find(key) != seenLookNames.end()) continue;
      LookPresetValues parsed{};
      bool parsedOk = false;
      if (!payload.empty()) parsedOk = parseLookValues(payload, &parsed);
      if (!parsedOk && !namedValues.empty()) parsedOk = parseLookValuesFromNamedJson(namedValues, &parsed);
      if (!parsedOk) continue;
      UserLookPreset p{};
      p.id = id; p.name = name; p.createdAtUtc = created.empty() ? nowUtcIso8601() : created; p.updatedAtUtc = updated.empty() ? p.createdAtUtc : updated; p.values = parsed;
      s.lookPresets.push_back(p);
      seenLookNames[key] = true;
    } else if (sec == Section::Tone) {
      const std::string key = normalizePresetNameKey(name);
      if (seenToneNames.find(key) != seenToneNames.end()) continue;
      TonescalePresetValues parsed{};
      bool parsedOk = false;
      if (!payload.empty()) parsedOk = parseTonescaleValues(payload, &parsed);
      if (!parsedOk && !namedValues.empty()) parsedOk = parseTonescaleValuesFromNamedJson(namedValues, &parsed);
      if (!parsedOk) continue;
      UserTonescalePreset p{};
      p.id = id; p.name = name; p.createdAtUtc = created.empty() ? nowUtcIso8601() : created; p.updatedAtUtc = updated.empty() ? p.createdAtUtc : updated; p.values = parsed;
      s.tonescalePresets.push_back(p);
      seenToneNames[key] = true;
    }
  }
}

int findUserLookIndexByNameLocked(const std::string& name) {
  const std::string n = normalizePresetNameKey(name);
  auto& v = userPresetStore().lookPresets;
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    if (normalizePresetNameKey(v[static_cast<size_t>(i)].name) == n) return i;
  }
  return -1;
}

int findUserTonescaleIndexByNameLocked(const std::string& name) {
  const std::string n = normalizePresetNameKey(name);
  auto& v = userPresetStore().tonescalePresets;
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    if (normalizePresetNameKey(v[static_cast<size_t>(i)].name) == n) return i;
  }
  return -1;
}

bool lookNameExistsLocked(const std::string& name, const std::string* ignoreId = nullptr) {
  const std::string key = normalizePresetNameKey(name);
  for (const char* builtIn : kLookPresetNames) {
    if (normalizePresetNameKey(builtIn) == key) return true;
  }
  for (const auto& p : userPresetStore().lookPresets) {
    if (ignoreId && !ignoreId->empty() && p.id == *ignoreId) continue;
    if (normalizePresetNameKey(p.name) == key) return true;
  }
  return false;
}

bool tonescaleNameExistsLocked(const std::string& name, const std::string* ignoreId = nullptr) {
  const std::string key = normalizePresetNameKey(name);
  for (const char* builtIn : kTonescalePresetNames) {
    if (normalizePresetNameKey(builtIn) == key) return true;
  }
  for (const auto& p : userPresetStore().tonescalePresets) {
    if (ignoreId && !ignoreId->empty() && p.id == *ignoreId) continue;
    if (normalizePresetNameKey(p.name) == key) return true;
  }
  return false;
}

// Force a full in-memory reload from disk. Used by explicit Refresh and menu resync paths.
void reloadUserPresetStoreFromDiskLocked() {
  UserPresetStore& s = userPresetStore();
  s = UserPresetStore{};
  ensureUserPresetStoreLoadedLocked();
}

bool userLookIndexFromPresetIndex(int idx, int* out) {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  const int rel = idx - kBuiltInLookPresetCount;
  if (rel < 0 || rel >= static_cast<int>(userPresetStore().lookPresets.size())) return false;
  *out = rel;
  return true;
}

int presetIndexFromUserLookIndex(int i) {
  if (i < 0) return -1;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  if (i >= static_cast<int>(userPresetStore().lookPresets.size())) return -1;
  return kBuiltInLookPresetCount + i;
}

bool isUserLookPresetIndex(int idx) {
  int i = -1;
  return userLookIndexFromPresetIndex(idx, &i);
}

bool userTonescaleIndexFromPresetIndex(int idx, int* out) {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  const int rel = idx - kBuiltInTonescalePresetCount;
  if (rel < 0 || rel >= static_cast<int>(userPresetStore().tonescalePresets.size())) return false;
  *out = rel;
  return true;
}

int presetIndexFromUserTonescaleIndex(int i) {
  if (i < 0) return -1;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  if (i >= static_cast<int>(userPresetStore().tonescalePresets.size())) return -1;
  return kBuiltInTonescalePresetCount + i;
}

bool isUserTonescalePresetIndex(int idx) {
  int i = -1;
  return userTonescaleIndexFromPresetIndex(idx, &i);
}

// ===== Preset <-> Menu Index Mapping and visible user-name lists =====
std::vector<std::string> visibleUserLookNames() {
  std::vector<std::string> out;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  for (const auto& p : userPresetStore().lookPresets) out.push_back(p.name);
  return out;
}

std::vector<std::string> visibleUserTonescaleNames() {
  std::vector<std::string> out;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  for (const auto& p : userPresetStore().tonescalePresets) out.push_back(p.name);
  return out;
}

// ===== Preset Application Helpers: resolved params and live OFX param writes =====
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

void writeLookValuesToParams(const LookPresetValues& s, OFX::ImageEffect& fx) {
  setDoubleIfPresent(fx, "tn_con", s.tn_con);
  setDoubleIfPresent(fx, "tn_sh", s.tn_sh);
  setDoubleIfPresent(fx, "tn_toe", s.tn_toe);
  setDoubleIfPresent(fx, "tn_off", s.tn_off);
  setBoolIfPresent(fx, "tn_hcon_enable", s.tn_hcon_enable != 0);
  setDoubleIfPresent(fx, "tn_hcon", s.tn_hcon);
  setDoubleIfPresent(fx, "tn_hcon_pv", s.tn_hcon_pv);
  setDoubleIfPresent(fx, "tn_hcon_st", s.tn_hcon_st);
  setBoolIfPresent(fx, "tn_lcon_enable", s.tn_lcon_enable != 0);
  setDoubleIfPresent(fx, "tn_lcon", s.tn_lcon);
  setDoubleIfPresent(fx, "tn_lcon_w", s.tn_lcon_w);
  setDoubleIfPresent(fx, "rs_sa", s.rs_sa);
  setDoubleIfPresent(fx, "rs_rw", s.rs_rw);
  setDoubleIfPresent(fx, "rs_bw", s.rs_bw);
  setBoolIfPresent(fx, "pt_enable", s.pt_enable != 0);
  setDoubleIfPresent(fx, "pt_lml", s.pt_lml);
  setDoubleIfPresent(fx, "pt_lml_r", s.pt_lml_r);
  setDoubleIfPresent(fx, "pt_lml_g", s.pt_lml_g);
  setDoubleIfPresent(fx, "pt_lml_b", s.pt_lml_b);
  setDoubleIfPresent(fx, "pt_lmh", s.pt_lmh);
  setDoubleIfPresent(fx, "pt_lmh_r", s.pt_lmh_r);
  setDoubleIfPresent(fx, "pt_lmh_b", s.pt_lmh_b);
  setBoolIfPresent(fx, "ptl_enable", s.ptl_enable != 0);
  setDoubleIfPresent(fx, "ptl_c", s.ptl_c);
  setDoubleIfPresent(fx, "ptl_m", s.ptl_m);
  setDoubleIfPresent(fx, "ptl_y", s.ptl_y);
  setBoolIfPresent(fx, "ptm_enable", s.ptm_enable != 0);
  setDoubleIfPresent(fx, "ptm_low", s.ptm_low);
  setDoubleIfPresent(fx, "ptm_low_rng", s.ptm_low_rng);
  setDoubleIfPresent(fx, "ptm_low_st", s.ptm_low_st);
  setDoubleIfPresent(fx, "ptm_high", s.ptm_high);
  setDoubleIfPresent(fx, "ptm_high_rng", s.ptm_high_rng);
  setDoubleIfPresent(fx, "ptm_high_st", s.ptm_high_st);
  setBoolIfPresent(fx, "brl_enable", s.brl_enable != 0);
  setDoubleIfPresent(fx, "brl", s.brl);
  setDoubleIfPresent(fx, "brl_r", s.brl_r);
  setDoubleIfPresent(fx, "brl_g", s.brl_g);
  setDoubleIfPresent(fx, "brl_b", s.brl_b);
  setDoubleIfPresent(fx, "brl_rng", s.brl_rng);
  setDoubleIfPresent(fx, "brl_st", s.brl_st);
  setBoolIfPresent(fx, "brlp_enable", s.brlp_enable != 0);
  setDoubleIfPresent(fx, "brlp", s.brlp);
  setDoubleIfPresent(fx, "brlp_r", s.brlp_r);
  setDoubleIfPresent(fx, "brlp_g", s.brlp_g);
  setDoubleIfPresent(fx, "brlp_b", s.brlp_b);
  setBoolIfPresent(fx, "hc_enable", s.hc_enable != 0);
  setDoubleIfPresent(fx, "hc_r", s.hc_r);
  setDoubleIfPresent(fx, "hc_r_rng", s.hc_r_rng);
  setBoolIfPresent(fx, "hs_rgb_enable", s.hs_rgb_enable != 0);
  setDoubleIfPresent(fx, "hs_r", s.hs_r);
  setDoubleIfPresent(fx, "hs_r_rng", s.hs_r_rng);
  setDoubleIfPresent(fx, "hs_g", s.hs_g);
  setDoubleIfPresent(fx, "hs_g_rng", s.hs_g_rng);
  setDoubleIfPresent(fx, "hs_b", s.hs_b);
  setDoubleIfPresent(fx, "hs_b_rng", s.hs_b_rng);
  setBoolIfPresent(fx, "hs_cmy_enable", s.hs_cmy_enable != 0);
  setDoubleIfPresent(fx, "hs_c", s.hs_c);
  setDoubleIfPresent(fx, "hs_c_rng", s.hs_c_rng);
  setDoubleIfPresent(fx, "hs_m", s.hs_m);
  setDoubleIfPresent(fx, "hs_m_rng", s.hs_m_rng);
  setDoubleIfPresent(fx, "hs_y", s.hs_y);
  setDoubleIfPresent(fx, "hs_y_rng", s.hs_y_rng);
  setIntIfPresent(fx, "cwp", s.cwp);
  setDoubleIfPresent(fx, "cwp_lm", s.cwp_lm);
}

void writeTonescaleValuesToParams(const TonescalePresetValues& t, OFX::ImageEffect& fx) {
  setDoubleIfPresent(fx, "tn_con", t.tn_con);
  setDoubleIfPresent(fx, "tn_sh", t.tn_sh);
  setDoubleIfPresent(fx, "tn_toe", t.tn_toe);
  setDoubleIfPresent(fx, "tn_off", t.tn_off);
  setBoolIfPresent(fx, "tn_hcon_enable", t.tn_hcon_enable != 0);
  setDoubleIfPresent(fx, "tn_hcon", t.tn_hcon);
  setDoubleIfPresent(fx, "tn_hcon_pv", t.tn_hcon_pv);
  setDoubleIfPresent(fx, "tn_hcon_st", t.tn_hcon_st);
  setBoolIfPresent(fx, "tn_lcon_enable", t.tn_lcon_enable != 0);
  setDoubleIfPresent(fx, "tn_lcon", t.tn_lcon);
  setDoubleIfPresent(fx, "tn_lcon_w", t.tn_lcon_w);
}

// ===== UI Text: Parameter tooltip lookup table =====
const char* tooltipForParam(const std::string& name) {
  static const std::unordered_map<std::string, const char*> kTooltips = {
    {"tn_Lp", "Peak display luminance target in nits."},
    {"tn_gb", "Amount of stops to boost grey luminance per stop of peak luminance increase."},
    {"pt_hdr", "How much purity compression and hue shift behavior changes as peak luminance increases."},
    {"tn_Lg", "Display luminance target for middle grey (0.18) in nits."},
    {"lookPreset", "Select a look preset. This applies look controls, while independent tonescale and creative-white selections are preserved."},
    {"tonescalePreset", "Select a tonescale preset, or use the current look preset tonescale with 'USE LOOK PRESET'."},
    {"creativeWhitePreset", "Select creative whitepoint behavior. 'USE LOOK PRESET' follows the selected look baseline."},
    {"cwp_lm", "Limit the intensity range affected by Creative Whitepoint."},
    {"displayEncodingPreset", "Choose a target viewing environment preset (EOTF, gamut, surround, and clamp defaults)."},
    {"tn_con", "Tonescale contrast (display-linear slope control)."},
    {"tn_sh", "Tonescale shoulder control; affects where highlights approach peak and clip."},
    {"tn_toe", "Shadow toe compression amount."},
    {"tn_off", "Scene-linear offset applied before tonescale."},
    {"tn_hcon_enable", "Enable highlight contrast shaping."},
    {"tn_hcon", "Highlight contrast amount."},
    {"tn_hcon_pv", "Stops above middle grey where highlight adjustment starts."},
    {"tn_hcon_st", "Highlight contrast transition strength/ramp-in speed."},
    {"tn_lcon_enable", "Enable low/mid contrast shaping."},
    {"tn_lcon", "Low-contrast module amount."},
    {"tn_lcon_w", "Low-contrast width (how broad the affected range is)."},
    {"rs_sa", "Render-space color-contrast amount (desaturates toward weighted luminance axis)."},
    {"rs_rw", "Render-space red weight."},
    {"rs_bw", "Render-space blue weight."},
    {"pt_enable", "Enable purity compression at higher intensities."},
    {"pt_lml", "Purity compression limit as intensity decreases (all hues)."},
    {"pt_lml_r", "Purity compression limit as intensity decreases (reds)."},
    {"pt_lml_g", "Purity compression limit as intensity decreases (greens)."},
    {"pt_lml_b", "Purity compression limit as intensity decreases (blues)."},
    {"pt_lmh", "Purity compression limit as intensity increases (all hues)."},
    {"pt_lmh_r", "Purity compression limit as intensity increases (reds)."},
    {"pt_lmh_b", "Purity compression limit as intensity increases (blues)."},
    {"ptl_enable", "Enable purity softclip to reduce hard clipping near gamut boundaries."},
    {"ptl_c", "Purity softclip strength for cyan."},
    {"ptl_m", "Purity softclip strength for magenta."},
    {"ptl_y", "Purity softclip strength for yellow."},
    {"ptm_enable", "Enable mid-purity shaping controls."},
    {"ptm_low", "Amount to raise mid-purity in low/mid intensity regions."},
    {"ptm_low_rng", "Range for low/mid mid-purity shaping."},
    {"ptm_low_st", "Strength weighting for low/mid mid-purity shaping."},
    {"ptm_high", "Amount to reduce mid-purity in upper-mid/high intensity regions."},
    {"ptm_high_rng", "Range for high mid-purity shaping."},
    {"ptm_high_st", "Strength weighting for high mid-purity shaping."},
    {"brl_enable", "Enable Brilliance (pre-compression purity-intensity shaping)."},
    {"brl", "Global brilliance amount for high-purity stimuli."},
    {"brl_r", "Brilliance amount for red stimuli."},
    {"brl_g", "Brilliance amount for green stimuli."},
    {"brl_b", "Brilliance amount for blue stimuli."},
    {"brl_rng", "Brilliance range over intensity (higher values affect lower intensities more)."},
    {"brl_st", "Brilliance strength over purity (higher values affect lower purity more)."},
    {"brlp_enable", "Enable Post Brilliance (after purity compression/hue shifts)."},
    {"brlp", "Global post-brilliance amount."},
    {"brlp_r", "Post-brilliance amount for red; useful to reduce high-purity ringing/halos."},
    {"brlp_g", "Post-brilliance amount for green; useful to reduce high-purity ringing/halos."},
    {"brlp_b", "Post-brilliance amount for blue; useful to reduce high-purity ringing/halos."},
    {"hc_enable", "Enable Hue Contrast shaping."},
    {"hc_r", "Hue contrast amount at red hue angle."},
    {"hc_r_rng", "Hue contrast range control over intensity."},
    {"hs_rgb_enable", "Enable RGB hue-shift module (primary hue-angle distortion as intensity increases)."},
    {"hs_r", "Red hue-shift amount (toward yellow)."},
    {"hs_r_rng", "Range of red hue-shift."},
    {"hs_g", "Green hue-shift amount (toward yellow)."},
    {"hs_g_rng", "Range of green hue-shift."},
    {"hs_b", "Blue hue-shift amount (toward cyan)."},
    {"hs_b_rng", "Range of blue hue-shift."},
    {"hs_cmy_enable", "Enable CMY hue-shift module (secondary hue-angle distortion as intensity decreases)."},
    {"hs_c", "Cyan hue-shift amount (toward blue)."},
    {"hs_c_rng", "Range of cyan hue-shift."},
    {"hs_m", "Magenta hue-shift amount (toward blue)."},
    {"hs_m_rng", "Range of magenta hue-shift."},
    {"hs_y", "Yellow hue-shift amount (toward red)."},
    {"hs_y_rng", "Range of yellow hue-shift."},
    {"clamp", "Clamp final image to display-supported range."},
    {"tn_su", "Surround compensation mode (dark, dim, bright)."},
    {"display_gamut", "Target display gamut."},
    {"eotf", "Target display transfer function."},
    {"crv_enable", "Draw tonescale overlay for curve debugging/inspection."}
  };
  auto it = kTooltips.find(name);
  return it == kTooltips.end() ? nullptr : it->second;
}

class OpenDRTEffect : public OFX::ImageEffect {
 public:
  // ===== Effect Lifecycle: construct + initial menu/state sync =====
  explicit OpenDRTEffect(OfxImageEffectHandle handle)
      : ImageEffect(handle) {
    dstClip_ = fetchClip(kOfxImageEffectOutputClipName);
    srcClip_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    suppressParamChanged_ = true;
    syncPresetMenusFromDisk(0.0, getChoice("lookPreset", 0.0, 0), getChoice("tonescalePreset", 0.0, 0));
    suppressParamChanged_ = false;
    updateToggleVisibility(0.0);
    updatePresetManagerActionState(0.0);
    updateReadonlyDisplayLabels(0.0);
  }

  ~OpenDRTEffect() override {
#if defined(OFX_SUPPORTS_CUDARENDER)
    if (stageSrcPinned_ != nullptr) {
      cudaFreeHost(stageSrcPinned_);
      stageSrcPinned_ = nullptr;
    }
    if (stageDstPinned_ != nullptr) {
      cudaFreeHost(stageDstPinned_);
      stageDstPinned_ = nullptr;
    }
    stagePinnedCapacityFloats_ = 0;
#endif
  }

  // ===== Render Path Entry =====
  // Main render callback.
  // Rule: keep preset/file management out of this path for predictable playback.
  void render(const OFX::RenderArguments& args) override {
    const auto tRenderStart = std::chrono::steady_clock::now();
    std::unique_ptr<OFX::Image> src(srcClip_->fetchImage(args.time));
    std::unique_ptr<OFX::Image> dst(dstClip_->fetchImage(args.time));

    if (!src || !dst) {
      OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    if (src->getPixelDepth() != OFX::eBitDepthFloat || dst->getPixelDepth() != OFX::eBitDepthFloat ||
        src->getPixelComponents() != OFX::ePixelComponentRGBA || dst->getPixelComponents() != OFX::ePixelComponentRGBA) {
      OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }

    const OfxRectI bounds = dst->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    if (width <= 0 || height <= 0) {
      return;
    }

    const size_t rowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
    struct RowLayout {
      bool valid = false;
      bool contiguous = false;
      float* base = nullptr;
      size_t pitchBytes = 0;
    };
    // Detect host row layout so we can use the direct path when rows are contiguous.
    auto detectLayout = [&](OFX::Image* img) -> RowLayout {
      RowLayout out{};
      out.base = static_cast<float*>(img->getPixelAddress(bounds.x1, bounds.y1));
      if (out.base == nullptr) return out;
      if (height <= 1) {
        out.valid = true;
        out.contiguous = true;
        out.pitchBytes = rowBytes;
        return out;
      }
      const char* prev = reinterpret_cast<const char*>(out.base);
      std::ptrdiff_t step = 0;
      for (int y = bounds.y1 + 1; y < bounds.y2; ++y) {
        float* row = static_cast<float*>(img->getPixelAddress(bounds.x1, y));
        if (row == nullptr) return RowLayout{};
        const char* cur = reinterpret_cast<const char*>(row);
        if (y == bounds.y1 + 1) {
          step = cur - prev;
        } else if (cur - prev != step) {
          return RowLayout{};
        }
        prev = cur;
      }
      out.valid = true;
      out.pitchBytes = static_cast<size_t>(step);
      out.contiguous = (out.pitchBytes == rowBytes);
      return out;
    };
    const RowLayout srcLayout = detectLayout(src.get());
    const RowLayout dstLayout = detectLayout(dst.get());

    const auto tResolveStart = std::chrono::steady_clock::now();
    OpenDRTRawValues raw = readRawValues(args.time);
    OpenDRTParams params = resolveParams(raw);
    perfLog("Param resolve", tResolveStart);

    if (!processor_) {
      processor_ = std::make_unique<OpenDRTProcessor>(params);
    } else {
      processor_->setParams(params);
    }

#if defined(OFX_SUPPORTS_CUDARENDER)
    // Optional OFX host CUDA mode:
    // - Controlled by selectedCudaRenderMode().
    // - Uses host-provided CUDA stream and device pointers from fetchImage().
    // - Avoids host<->device staging copies.
    // Note-to-self:
    // This is the fastest route for playback. If I see "Backend render direct"
    // in logs on a CUDA-enabled host, this branch was not taken.
    const bool preferHostCuda = (selectedCudaRenderMode() == CudaRenderMode::HostPreferred);
    const bool tryHostCuda = preferHostCuda && args.isEnabledCudaRender && (args.pCudaStream != nullptr);
    if (tryHostCuda) {
      const auto tHostCuda = std::chrono::steady_clock::now();
      const float* srcDevice = static_cast<const float*>(src->getPixelData());
      float* dstDevice = static_cast<float*>(dst->getPixelData());
      const int srcRb = src->getRowBytes();
      const int dstRb = dst->getRowBytes();
      const size_t srcRowBytes = srcRb < 0 ? static_cast<size_t>(-srcRb) : static_cast<size_t>(srcRb);
      const size_t dstRowBytes = dstRb < 0 ? static_cast<size_t>(-dstRb) : static_cast<size_t>(dstRb);
      if (srcDevice != nullptr && dstDevice != nullptr &&
          processor_->renderCUDAHostBuffers(srcDevice, dstDevice, width, height, srcRowBytes, dstRowBytes, args.pCudaStream)) {
        perfLog("Backend render host CUDA", tHostCuda);
        perfLog("Render total", tRenderStart);
        return;
      }
      if (debugLogEnabled()) {
        std::fprintf(stderr, "[ME_OpenDRT] Host CUDA render failed.\n");
      }
      // When the host explicitly provided CUDA memory, do not fall through into CPU staging
      // paths that assume host-readable pointers.
      OFX::throwSuiteStatusException(kOfxStatFailed);
    }
#endif

#if defined(__APPLE__)
    // Host Metal mode (macOS):
    // - Uses host-provided command queue + MTLBuffer image handles.
    // - Avoids plugin-owned CPU staging copies.
    const bool preferHostMetal = (selectedMetalRenderMode() == MetalRenderMode::HostPreferred);
    const bool tryHostMetal = preferHostMetal && args.isEnabledMetalRender && (args.pMetalCmdQ != nullptr);
    if (tryHostMetal) {
      const auto tHostMetal = std::chrono::steady_clock::now();
      const void* srcMetalBuffer = src->getPixelData();
      void* dstMetalBuffer = dst->getPixelData();
      const int srcRb = src->getRowBytes();
      const int dstRb = dst->getRowBytes();
      const size_t srcRowBytes = srcRb < 0 ? static_cast<size_t>(-srcRb) : static_cast<size_t>(srcRb);
      const size_t dstRowBytes = dstRb < 0 ? static_cast<size_t>(-dstRb) : static_cast<size_t>(dstRb);
      if (srcMetalBuffer != nullptr && dstMetalBuffer != nullptr &&
          processor_->renderMetalHostBuffers(
              srcMetalBuffer, dstMetalBuffer, width, height, srcRowBytes, dstRowBytes, args.pMetalCmdQ)) {
        OpenDRTMetal::resetHostMetalFailureState();
        perfLog("Backend render host Metal", tHostMetal);
        perfLog("Render total", tRenderStart);
        return;
      }
      if (debugLogEnabled()) {
        std::fprintf(stderr, "[ME_OpenDRT] Host Metal render failed.\n");
      }
      // Safe fallback: continue into existing internal render path.
      // This preserves stability if host-Metal submission fails transiently.
    }
#endif

    bool rendered = false;
    // Fast path: process directly on host image memory layout (no extra staging vectors).
    if (!forceStageCopyEnabled() && srcLayout.valid && dstLayout.valid) {
      const auto tBackendDirect = std::chrono::steady_clock::now();
      rendered = processor_->renderWithLayout(
          srcLayout.base, dstLayout.base, width, height, srcLayout.pitchBytes, dstLayout.pitchBytes, true, false);
      perfLog("Backend render direct", tBackendDirect);
    }

    // Fallback path: stable staged copy used for irregular host layouts.
    if (!rendered) {
      const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
      if (!ensureStageBuffers(pixelCount)) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
      }
      float* srcStage = stageSrcPtr();
      float* dstStage = stageDstPtr();
      if (!srcStage || !dstStage) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
      }

      const auto tStageCopyStart = std::chrono::steady_clock::now();
      if (srcLayout.valid && srcLayout.contiguous) {
        std::memcpy(srcStage, srcLayout.base, rowBytes * static_cast<size_t>(height));
      } else {
        // Row fallback for hosts with non-contiguous row layout.
        for (int y = bounds.y1; y < bounds.y2; ++y) {
          const int localY = y - bounds.y1;
          float* sp = static_cast<float*>(src->getPixelAddress(bounds.x1, y));
          float* rowDst = srcStage + static_cast<size_t>(localY) * static_cast<size_t>(width) * 4u;
          if (sp != nullptr) {
            std::memcpy(rowDst, sp, rowBytes);
          } else {
            std::memset(rowDst, 0, rowBytes);
          }
        }
      }
      perfLog("Host src staging", tStageCopyStart);

      const auto tBackendStart = std::chrono::steady_clock::now();
      rendered = processor_->render(srcStage, dstStage, width, height, true, false);
      perfLog("Backend render staging", tBackendStart);
      if (!rendered) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
      }

      const auto tDstCopyStart = std::chrono::steady_clock::now();
      if (dstLayout.valid && dstLayout.contiguous) {
        std::memcpy(dstLayout.base, dstStage, rowBytes * static_cast<size_t>(height));
      } else {
        for (int y = bounds.y1; y < bounds.y2; ++y) {
          const int localY = y - bounds.y1;
          float* dp = static_cast<float*>(dst->getPixelAddress(bounds.x1, y));
          if (!dp) continue;
          const float* rowSrc = dstStage + static_cast<size_t>(localY) * static_cast<size_t>(width) * 4u;
          std::memcpy(dp, rowSrc, rowBytes);
        }
      }
      perfLog("Host dst copy", tDstCopyStart);
    }

    perfLog("Render total", tRenderStart);
  }

  // ===== UI Event Entry =====
  // UI/param callback entry point.
  // Keep this deterministic: mutate params/state, then refresh dependent UI labels/states.
  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override {
    try {
      if (suppressParamChanged_) {
        return;
      }
      if (args.reason == OFX::eChangePluginEdit || args.reason == OFX::eChangeTime) {
        return;
      }

      if (paramName == "presetState") {
        return;
      }
      if (paramName == "activeUserLookSlot" || paramName == "activeUserToneSlot") {
        return;
      }

      // ----- Preset Routing: Look selector -----
      // Look preset selection updates look-driven controls.
      // Tonescale/CWP selectors explicitly chosen by the user are preserved.
      if (paramName == "lookPreset") {
        // Intent:
        // - preserve user overrides when they explicitly selected non-zero preset choices
        // - still apply new look defaults for all other look-driven controls
        // Why:
        // Previous behavior always forced tonescale/CWP back to "USE LOOK PRESET", which
        // unexpectedly discarded user choices while browsing looks.
        int look = getChoice("lookPreset", args.time, 0);
        const int tsPreset = getChoice("tonescalePreset", args.time, 0);
        const int cwpPreset = getChoice("creativeWhitePreset", args.time, 0);
        const bool preserveTonescale = (tsPreset != 0);
        const bool preserveCwp = (cwpPreset > 0);
        const TonescalePresetValues preservedTs = preserveTonescale ? captureCurrentTonescaleValues(args.time) : TonescalePresetValues{};
        const int preservedCwp = preserveCwp ? getInt("cwp", args.time, 2) : 2;
        int activeToneUser = -1;
        if (isUserTonescalePresetIndex(tsPreset)) {
          int userToneIdx = -1;
          if (userTonescaleIndexFromPresetIndex(tsPreset, &userToneIdx)) activeToneUser = userToneIdx;
        }
        FlagScope scope(suppressParamChanged_);
        // active user slots track menu identity; this keeps update/delete/rename actions aimed
        // at the correct user preset after a look change.
        setInt("activeUserLookSlot", -1);
        setInt("activeUserToneSlot", activeToneUser);
        if (isUserLookPresetIndex(look)) {
          int userIdx = -1;
          if (!userLookIndexFromPresetIndex(look, &userIdx)) return;
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          const auto& userPreset = userPresetStore().lookPresets[static_cast<size_t>(userIdx)];
          writeLookValuesToParams(userPreset.values, *this);
          setInt("activeUserLookSlot", userIdx);
        } else {
          writePresetToParams(look, *this);
        }
        if (preserveTonescale) {
          // Re-apply captured tonescale values after look write, so look switch does not stomp
          // an explicit tonescale preset/user tweak.
          writeTonescaleValuesToParams(preservedTs, *this);
        }
        if (preserveCwp) {
          // Keep explicit creative-white override; cwp_lm intentionally remains look-driven.
          setInt("cwp", preservedCwp);
        }
        updateToggleVisibility(args.time);
        updatePresetManagerActionState(args.time);
        updateReadonlyDisplayLabels(args.time);
        updatePresetStateFromCurrent(args.time);
        return;
      }

      // ----- Preset Routing: Tonescale selector -----
      // Tonescale preset can be independent, or inherit from currently selected look when index 0 is chosen.
      if (paramName == "tonescalePreset") {
        const int tsPreset = getChoice("tonescalePreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        setInt("activeUserToneSlot", -1);
        if (isUserTonescalePresetIndex(tsPreset)) {
          int userIdx = -1;
          if (!userTonescaleIndexFromPresetIndex(tsPreset, &userIdx)) return;
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          const auto& userPreset = userPresetStore().tonescalePresets[static_cast<size_t>(userIdx)];
          writeTonescaleValuesToParams(userPreset.values, *this);
          setInt("activeUserToneSlot", userIdx);
        } else if (tsPreset == 0) {
          const TonescalePresetValues fromLook = selectedLookBaseTonescale(args.time);
          writeTonescaleValuesToParams(fromLook, *this);
        } else {
          writeTonescalePresetToParams(tsPreset, *this);
        }
        updateToggleVisibility(args.time);
        updatePresetManagerActionState(args.time);
        updatePresetStateFromCurrent(args.time);
        updateReadonlyDisplayLabels(args.time);
        return;
      }

      // ----- Preset Routing: Creative white selector -----
      if (paramName == "creativeWhitePreset") {
        const int cwpPreset = getChoice("creativeWhitePreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        if (cwpPreset <= 0) {
          setInt("cwp", selectedLookBaseCwp(args.time));
        } else {
          writeCreativeWhitePresetToParams(cwpPreset, *this);
        }
        updatePresetStateFromCurrent(args.time);
        updateReadonlyDisplayLabels(args.time);
        return;
      }

      // ----- Preset Routing: Display encoding selector -----
      if (paramName == "displayEncodingPreset") {
        int preset = getChoice("displayEncodingPreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        writeDisplayPresetToParams(preset, *this);
        updatePresetStateFromCurrent(args.time);
        updateReadonlyDisplayLabels(args.time);
        return;
      }

      // ----- Preset State: Discard all modifications to current baseline -----
      if (paramName == "discardPresetChanges") {
        OpenDRTParams expected{};
        if (!buildPresetBaseline(args.time, &expected)) return;
        FlagScope scope(suppressParamChanged_);
        applyTonescaleFromBaseline(expected);
        applyRenderSpaceFromBaseline(expected);
        applyMidPurityFromBaseline(expected);
        applyPurityCompressionFromBaseline(expected);
        applyBrillianceFromBaseline(expected);
        applyHueFromBaseline(expected);
        setBool("clamp", expected.clamp != 0);
        setChoice("tn_su", expected.tn_su);
        setChoice("display_gamut", expected.display_gamut);
        setChoice("eotf", expected.eotf);
        setInt("cwp", expected.cwp);
        setDouble("cwp_lm", expected.cwp_lm);
        updateToggleVisibility(args.time);
        updatePresetManagerActionState(args.time);
        updateReadonlyDisplayLabels(args.time);
        updatePresetStateFromCurrent(args.time);
        return;
      }

      // ----- Preset State: Per-category reset buttons -----
      if (paramName == "reset_tonescale" ||
          paramName == "reset_render_space" ||
          paramName == "reset_mid_purity" ||
          paramName == "reset_purity_compression" ||
          paramName == "reset_brilliance" ||
          paramName == "reset_hue") {
        OpenDRTParams expected{};
        if (!buildPresetBaseline(args.time, &expected)) return;
        FlagScope scope(suppressParamChanged_);
        if (paramName == "reset_tonescale") {
          applyTonescaleFromBaseline(expected);
        } else if (paramName == "reset_render_space") {
          applyRenderSpaceFromBaseline(expected);
        } else if (paramName == "reset_mid_purity") {
          applyMidPurityFromBaseline(expected);
        } else if (paramName == "reset_purity_compression") {
          applyPurityCompressionFromBaseline(expected);
        } else if (paramName == "reset_brilliance") {
          applyBrillianceFromBaseline(expected);
        } else if (paramName == "reset_hue") {
          applyHueFromBaseline(expected);
        }
        updateToggleVisibility(args.time);
        updatePresetManagerActionState(args.time);
        updateReadonlyDisplayLabels(args.time);
        updatePresetStateFromCurrent(args.time);
        return;
      }

      // Support actions are side-effect free for grading state.
      if (paramName == "supportParametersGuide") {
        (void)openExternalUrl("https://github.com/jedypod/open-display-transform/blob/main/display-transforms/opendrt/docs/opendrt-parameters.md");
        return;
      }

      if (paramName == "supportLatestReleases") {
        (void)openExternalUrl("https://github.com/MoazElgabry/ME_OFX/releases");
        return;
      }

      if (paramName == "supportReportIssue") {
        (void)openExternalUrl("https://github.com/MoazElgabry/ME_OFX/issues");
        return;
      }

      if (paramName == "userLookSave") {
        const std::string name = sanitizePresetName(getString("userPresetName", "User Look"), "User Look");
        const LookPresetValues values = captureCurrentLookValues(args.time);
        int targetIndex = -1;
        {
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          if (lookNameExistsLocked(name)) {
            showInfoDialog("A Look preset with this name already exists.");
            return;
          }
          UserLookPreset p{};
          p.id = makePresetId("look");
          p.name = name;
          p.createdAtUtc = nowUtcIso8601();
          p.updatedAtUtc = p.createdAtUtc;
          p.values = values;
          userPresetStore().lookPresets.push_back(p);
          targetIndex = static_cast<int>(userPresetStore().lookPresets.size()) - 1;
          saveUserPresetStoreLocked();
        }
        FlagScope scope(suppressParamChanged_);
        setInt("activeUserLookSlot", targetIndex);
        const int idx = presetIndexFromUserLookIndex(targetIndex);
        syncPresetMenusFromDisk(args.time, idx >= 0 ? idx : 0, getChoice("tonescalePreset", args.time, 0));
        if (idx >= 0) setChoice("lookPreset", idx);
        writeLookValuesToParams(values, *this);
        updateToggleVisibility(args.time);
        return;
      }

      if (paramName == "userTonescaleSave") {
        const std::string name = sanitizePresetName(getString("userPresetName", "User Tonescale"), "User Tonescale");
        const TonescalePresetValues values = captureCurrentTonescaleValues(args.time);
        int targetIndex = -1;
        {
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          if (tonescaleNameExistsLocked(name)) {
            showInfoDialog("A Tonescale preset with this name already exists.");
            return;
          }
          UserTonescalePreset p{};
          p.id = makePresetId("tone");
          p.name = name;
          p.createdAtUtc = nowUtcIso8601();
          p.updatedAtUtc = p.createdAtUtc;
          p.values = values;
          userPresetStore().tonescalePresets.push_back(p);
          targetIndex = static_cast<int>(userPresetStore().tonescalePresets.size()) - 1;
          saveUserPresetStoreLocked();
        }
        FlagScope scope(suppressParamChanged_);
        setInt("activeUserToneSlot", targetIndex);
        const int idx = presetIndexFromUserTonescaleIndex(targetIndex);
        syncPresetMenusFromDisk(args.time, getChoice("lookPreset", args.time, 0), idx >= 0 ? idx : 0);
        if (idx >= 0) setChoice("tonescalePreset", idx);
        writeTonescaleValuesToParams(values, *this);
        updateToggleVisibility(args.time);
        return;
      }

      // ----- Preset Manager: Import -----
      // Import JSON presets into the local user library and apply imported values immediately.
      if (paramName == "userPresetImport") {
        const std::string path = pickOpenJsonFilePath();
        if (path.empty()) return;
        std::ifstream is(path, std::ios::binary);
        if (!is.is_open()) return;
        std::string content((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
        const std::string type = jsonField(content, "presetType");
        const std::string name = sanitizePresetName(jsonField(content, "name"), "Imported Preset");
        const std::string payload = jsonField(content, "payload");
        const std::string namedValues = jsonObjectField(content, "namedValues");
        if (type.empty() || (payload.empty() && namedValues.empty())) return;

        FlagScope scope(suppressParamChanged_);
        if (type == "look") {
          LookPresetValues values{};
          bool parsedOk = false;
          if (!payload.empty()) parsedOk = parseLookValues(payload, &values);
          if (!parsedOk && !namedValues.empty()) parsedOk = parseLookValuesFromNamedJson(namedValues, &values);
          if (!parsedOk) return;
          int index = -1;
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (lookNameExistsLocked(name)) {
              showInfoDialog("A Look preset with this name already exists.");
              return;
            }
            UserLookPreset p{};
            p.id = makePresetId("look"); p.name = name; p.createdAtUtc = nowUtcIso8601(); p.updatedAtUtc = p.createdAtUtc; p.values = values;
            userPresetStore().lookPresets.push_back(p);
            index = static_cast<int>(userPresetStore().lookPresets.size()) - 1;
            saveUserPresetStoreLocked();
          }
          const int idx = presetIndexFromUserLookIndex(index);
          syncPresetMenusFromDisk(args.time, idx >= 0 ? idx : 0, getChoice("tonescalePreset", args.time, 0));
          if (idx >= 0) setChoice("lookPreset", idx);
          setInt("activeUserLookSlot", index);
          writeLookValuesToParams(values, *this);
        } else if (type == "tonescale") {
          TonescalePresetValues values{};
          bool parsedOk = false;
          if (!payload.empty()) parsedOk = parseTonescaleValues(payload, &values);
          if (!parsedOk && !namedValues.empty()) parsedOk = parseTonescaleValuesFromNamedJson(namedValues, &values);
          if (!parsedOk) return;
          int index = -1;
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (tonescaleNameExistsLocked(name)) {
              showInfoDialog("A Tonescale preset with this name already exists.");
              return;
            }
            UserTonescalePreset p{};
            p.id = makePresetId("tone"); p.name = name; p.createdAtUtc = nowUtcIso8601(); p.updatedAtUtc = p.createdAtUtc; p.values = values;
            userPresetStore().tonescalePresets.push_back(p);
            index = static_cast<int>(userPresetStore().tonescalePresets.size()) - 1;
            saveUserPresetStoreLocked();
          }
          const int idx = presetIndexFromUserTonescaleIndex(index);
          syncPresetMenusFromDisk(args.time, getChoice("lookPreset", args.time, 0), idx >= 0 ? idx : 0);
          if (idx >= 0) setChoice("tonescalePreset", idx);
          setInt("activeUserToneSlot", index);
          writeTonescaleValuesToParams(values, *this);
        }
        updateToggleVisibility(args.time);
        return;
      }

      // ----- Preset Manager: Update selected user preset -----
      // Overwrite the currently selected user preset with current knob values.
      if (paramName == "userPresetUpdateCurrent") {
        const int lookIdx = getChoice("lookPreset", args.time, 0);
        const int toneIdx = getChoice("tonescalePreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        if (isUserLookPresetIndex(lookIdx)) {
          int userIdx = -1;
          if (!userLookIndexFromPresetIndex(lookIdx, &userIdx)) return;
          const LookPresetValues values = captureCurrentLookValues(args.time);
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return;
            auto& dst = userPresetStore().lookPresets[static_cast<size_t>(userIdx)];
            dst.values = values;
            dst.updatedAtUtc = nowUtcIso8601();
            saveUserPresetStoreLocked();
          }
          syncPresetMenusFromDisk(args.time, lookIdx, toneIdx);
          writeLookValuesToParams(values, *this);
          updatePresetStateFromCurrent(args.time);
          updateReadonlyDisplayLabels(args.time);
          return;
        }
        if (isUserTonescalePresetIndex(toneIdx)) {
          int userIdx = -1;
          if (!userTonescaleIndexFromPresetIndex(toneIdx, &userIdx)) return;
          const TonescalePresetValues values = captureCurrentTonescaleValues(args.time);
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().tonescalePresets.size())) return;
            auto& dst = userPresetStore().tonescalePresets[static_cast<size_t>(userIdx)];
            dst.values = values;
            dst.updatedAtUtc = nowUtcIso8601();
            saveUserPresetStoreLocked();
          }
          syncPresetMenusFromDisk(args.time, lookIdx, toneIdx);
          writeTonescaleValuesToParams(values, *this);
          updatePresetStateFromCurrent(args.time);
          updateReadonlyDisplayLabels(args.time);
          return;
        }
        updatePresetManagerActionState(args.time);
        return;
      }

      // ----- Preset Manager: Delete -----
      // Delete selected user preset(s) from disk-backed store and re-sync menu selections.
      if (paramName == "userPresetDeleteCurrent") {
        const int lookIdx = getChoice("lookPreset", args.time, 0);
        const int toneIdx = getChoice("tonescalePreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        const bool hasLookUser = isUserLookPresetIndex(lookIdx);
        const bool hasToneUser = isUserTonescalePresetIndex(toneIdx);
        if (!hasLookUser && !hasToneUser) {
          showInfoDialog("Select a user preset before deleting.");
          updatePresetManagerActionState(args.time);
          return;
        }

        DeleteTarget target = DeleteTarget::Cancel;
        if (hasLookUser && hasToneUser) {
          target = choosePresetTargetDialog("Delete");
          if (target == DeleteTarget::Cancel) return;
        } else {
          target = hasLookUser ? DeleteTarget::Look : DeleteTarget::Tonescale;
        }

        if (target == DeleteTarget::Look) {
          int userIdx = -1;
          if (!userLookIndexFromPresetIndex(lookIdx, &userIdx)) return;
          std::string presetName;
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return;
            presetName = userPresetStore().lookPresets[static_cast<size_t>(userIdx)].name;
          }
          if (!confirmDeleteDialog(presetName)) return;
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return;
            userPresetStore().lookPresets.erase(userPresetStore().lookPresets.begin() + userIdx);
            saveUserPresetStoreLocked();
          }
          syncPresetMenusFromDisk(args.time, 0, toneIdx);
          setChoice("lookPreset", 0);
          setInt("activeUserLookSlot", -1);
          return;
        }
        if (target == DeleteTarget::Tonescale) {
          int userIdx = -1;
          if (!userTonescaleIndexFromPresetIndex(toneIdx, &userIdx)) return;
          std::string presetName;
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().tonescalePresets.size())) return;
            presetName = userPresetStore().tonescalePresets[static_cast<size_t>(userIdx)].name;
          }
          if (!confirmDeleteDialog(presetName)) return;
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().tonescalePresets.size())) return;
            userPresetStore().tonescalePresets.erase(userPresetStore().tonescalePresets.begin() + userIdx);
            saveUserPresetStoreLocked();
          }
          syncPresetMenusFromDisk(args.time, lookIdx, 0);
          setChoice("tonescalePreset", 0);
          setInt("activeUserToneSlot", -1);
          return;
        }
        return;
      }

      // ----- Preset Manager: Refresh menus from disk -----
      // Re-scan preset file and rebuild menu options without changing non-preset controls.
      if (paramName == "userPresetRefresh") {
        FlagScope scope(suppressParamChanged_);
        syncPresetMenusFromDisk(args.time, getChoice("lookPreset", args.time, 0), getChoice("tonescalePreset", args.time, 0));
        return;
      }

      // ----- Preset Manager: Rename -----
      // Rename selected user preset, preserving its payload and identifier.
      if (paramName == "userPresetRenameCurrent") {
        const std::string newName = sanitizePresetName(getString("userPresetName", "User Preset"), "User Preset");
        const int lookIdx = getChoice("lookPreset", args.time, 0);
        const int toneIdx = getChoice("tonescalePreset", args.time, 0);
        const bool hasLookUser = isUserLookPresetIndex(lookIdx);
        const bool hasToneUser = isUserTonescalePresetIndex(toneIdx);
        FlagScope scope(suppressParamChanged_);
        if (!hasLookUser && !hasToneUser) {
          showInfoDialog("Select a user preset before renaming.");
          return;
        }

        DeleteTarget target = DeleteTarget::Cancel;
        if (hasLookUser && hasToneUser) {
          target = choosePresetTargetDialog("Rename");
          if (target == DeleteTarget::Cancel) return;
        } else {
          target = hasLookUser ? DeleteTarget::Look : DeleteTarget::Tonescale;
        }

        if (target == DeleteTarget::Look) {
          int userIdx = -1;
          if (!userLookIndexFromPresetIndex(lookIdx, &userIdx)) return;
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return;
            std::string ignoreId = userPresetStore().lookPresets[static_cast<size_t>(userIdx)].id;
            if (lookNameExistsLocked(newName, &ignoreId)) {
              showInfoDialog("A Look preset with this name already exists.");
              return;
            }
            auto& dst = userPresetStore().lookPresets[static_cast<size_t>(userIdx)];
            dst.name = newName;
            dst.updatedAtUtc = nowUtcIso8601();
            saveUserPresetStoreLocked();
          }
          syncPresetMenusFromDisk(args.time, lookIdx, toneIdx);
          setChoice("lookPreset", lookIdx);
          updateReadonlyDisplayLabels(args.time);
          return;
        }
        if (target == DeleteTarget::Tonescale) {
          int userIdx = -1;
          if (!userTonescaleIndexFromPresetIndex(toneIdx, &userIdx)) return;
          {
            std::lock_guard<std::mutex> lock(userPresetMutex());
            ensureUserPresetStoreLoadedLocked();
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().tonescalePresets.size())) return;
            std::string ignoreId = userPresetStore().tonescalePresets[static_cast<size_t>(userIdx)].id;
            if (tonescaleNameExistsLocked(newName, &ignoreId)) {
              showInfoDialog("A Tonescale preset with this name already exists.");
              return;
            }
            auto& dst = userPresetStore().tonescalePresets[static_cast<size_t>(userIdx)];
            dst.name = newName;
            dst.updatedAtUtc = nowUtcIso8601();
            saveUserPresetStoreLocked();
          }
          syncPresetMenusFromDisk(args.time, lookIdx, toneIdx);
          setChoice("tonescalePreset", toneIdx);
          updateReadonlyDisplayLabels(args.time);
          return;
        }
        return;
      }

      // ----- Preset Manager: Export -----
      // Export selected user preset as a portable JSON (canonical payload + named/nuke/dctl helpers).
      if (paramName == "userPresetExportLook" || paramName == "userPresetExportTonescale") {
        const bool exportLook = (paramName == "userPresetExportLook");
        const int lookIdx = getChoice("lookPreset", args.time, 0);
        const int toneIdx = getChoice("tonescalePreset", args.time, 0);
        std::string name;
        std::string type;
        std::string payload;
        std::string namedValues;
        std::string nukeCmd;
        std::string nukeMenuEntry;
        std::string dctlPreset;
        {
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          if (exportLook) {
            const int userIdx = lookIdx - kBuiltInLookPresetCount;
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return;
            const auto& p = userPresetStore().lookPresets[static_cast<size_t>(userIdx)];
            name = p.name;
            type = "look";
            serializeLookValues(p.values, payload);
            namedValues = lookValuesAsNamedJson(p.values);
            nukeCmd = lookValuesAsNukeCmd(name, p.values);
            nukeMenuEntry = std::string("Look Presets/") + name;
            dctlPreset = lookValuesAsDctl(p.values);
          } else {
            const int userIdx = toneIdx - kBuiltInTonescalePresetCount;
            if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().tonescalePresets.size())) return;
            const auto& p = userPresetStore().tonescalePresets[static_cast<size_t>(userIdx)];
            name = p.name;
            type = "tonescale";
            serializeTonescaleValues(p.values, payload);
            namedValues = tonescaleValuesAsNamedJson(p.values);
            nukeCmd = tonescaleValuesAsNukeCmd(p.values);
            nukeMenuEntry = std::string("Tonescale Presets/") + name;
            dctlPreset = tonescaleValuesAsDctl(p.values);
          }
        }
        const std::string file = pickSaveJsonFilePath(name + ".json");
        if (file.empty()) return;
        std::ofstream os(file, std::ios::binary | std::ios::trunc);
        if (!os.is_open()) return;
        os << "{\n";
        os << "  \"schemaVersion\":3,\n";
        os << "  \"presetType\":\"" << jsonEscape(type) << "\",\n";
        os << "  \"name\":\"" << jsonEscape(name) << "\",\n";
        os << "  \"payload\":\"" << jsonEscape(payload) << "\",\n";
        os << "  \"namedValues\":" << prettyJsonObject(namedValues, "  ") << ",\n";
        os << "  \"nuke\":{\n";
        os << "    \"menuEntry\":\"" << jsonEscape(nukeMenuEntry) << "\",\n";
        os << "    \"presetCmd\":\"" << jsonEscape(nukeCmd) << "\"\n";
        os << "  },\n";
        os << "  \"dctl\":{\n";
        os << "    \"preset\":\"" << jsonEscape(dctlPreset) << "\"\n";
        os << "  }\n";
        os << "}\n";
        return;
      }

      if (isAdvancedParam(paramName)) {
        FlagScope scope(suppressParamChanged_);
        if (isVisibilityToggleParam(paramName)) {
          updateToggleVisibility(args.time);
        }
        updatePresetStateFromCurrent(args.time);
        updateReadonlyDisplayLabels(args.time);
      }
    } catch (...) {
      // Swallow callback exceptions to avoid host crashes while stabilizing.
    }
  }

  void getClipPreferences(OFX::ClipPreferencesSetter& clipPreferences) override {
    clipPreferences.setClipBitDepth(*dstClip_, OFX::eBitDepthFloat);
    clipPreferences.setClipComponents(*dstClip_, OFX::ePixelComponentRGBA);
  }

 private:
  struct FlagScope {
    explicit FlagScope(bool& f) : flag(f) { flag = true; }
    ~FlagScope() { flag = false; }
    bool& flag;
  };

  // ===== Staging Buffers: host memory used by non-direct render paths =====
  bool ensureStageBuffers(size_t pixelCount) {
#if defined(OFX_SUPPORTS_CUDARENDER)
    // Prefer pinned host buffers for staged path to improve CUDA transfer throughput.
    if (stageSrcPinned_ != nullptr && stageDstPinned_ != nullptr && stagePinnedCapacityFloats_ == pixelCount) return true;
    if (stageSrcPinned_ != nullptr) {
      cudaFreeHost(stageSrcPinned_);
      stageSrcPinned_ = nullptr;
    }
    if (stageDstPinned_ != nullptr) {
      cudaFreeHost(stageDstPinned_);
      stageDstPinned_ = nullptr;
    }
    stagePinnedCapacityFloats_ = 0;
    const size_t bytes = pixelCount * sizeof(float);
    if (cudaHostAlloc(reinterpret_cast<void**>(&stageSrcPinned_), bytes, cudaHostAllocDefault) == cudaSuccess &&
        cudaHostAlloc(reinterpret_cast<void**>(&stageDstPinned_), bytes, cudaHostAllocDefault) == cudaSuccess) {
      stagePinnedCapacityFloats_ = pixelCount;
      return true;
    }
    if (stageSrcPinned_ != nullptr) {
      cudaFreeHost(stageSrcPinned_);
      stageSrcPinned_ = nullptr;
    }
    if (stageDstPinned_ != nullptr) {
      cudaFreeHost(stageDstPinned_);
      stageDstPinned_ = nullptr;
    }
    stagePinnedCapacityFloats_ = 0;
#endif
    if (srcPixels_.size() != pixelCount) srcPixels_.assign(pixelCount, 0.0f);
    if (dstPixels_.size() != pixelCount) dstPixels_.assign(pixelCount, 0.0f);
    return true;
  }

  float* stageSrcPtr() {
#if defined(OFX_SUPPORTS_CUDARENDER)
    if (stageSrcPinned_ != nullptr) return stageSrcPinned_;
#endif
    return srcPixels_.empty() ? nullptr : srcPixels_.data();
  }

  float* stageDstPtr() {
#if defined(OFX_SUPPORTS_CUDARENDER)
    if (stageDstPinned_ != nullptr) return stageDstPinned_;
#endif
    return dstPixels_.empty() ? nullptr : dstPixels_.data();
  }

  // ===== Param Classification: route updates and state recomputation =====
  bool isAdvancedParam(const std::string& name) const {
    static const std::vector<std::string> names = {
      "tn_con","tn_sh","tn_toe","tn_off","tn_hcon_enable","tn_hcon","tn_hcon_pv","tn_hcon_st","tn_lcon_enable","tn_lcon","tn_lcon_w",
      "rs_sa","rs_rw","rs_bw",
      "pt_enable","pt_lml","pt_lml_r","pt_lml_g","pt_lml_b","pt_lmh","pt_lmh_r","pt_lmh_b","ptl_enable","ptl_c","ptl_m","ptl_y",
      "ptm_enable","ptm_low","ptm_low_rng","ptm_low_st","ptm_high","ptm_high_rng","ptm_high_st",
      "brl_enable","brl","brl_r","brl_g","brl_b","brl_rng","brl_st","brlp_enable","brlp","brlp_r","brlp_g","brlp_b",
      "hc_enable","hc_r","hc_r_rng","hs_rgb_enable","hs_r","hs_r_rng","hs_g","hs_g_rng","hs_b","hs_b_rng","hs_cmy_enable","hs_c","hs_c_rng","hs_m","hs_m_rng","hs_y","hs_y_rng",
      "clamp","tn_su","display_gamut","eotf","cwp","cwp_lm"
    };
    for (const auto& n : names) if (n == name) return true;
    return false;
  }

  bool isTonescaleParam(const std::string& name) const {
    static const std::vector<std::string> names = {
      "tn_con","tn_sh","tn_toe","tn_off","tn_hcon_enable","tn_hcon","tn_hcon_pv","tn_hcon_st","tn_lcon_enable","tn_lcon","tn_lcon_w"
    };
    for (const auto& n : names) if (n == name) return true;
    return false;
  }

  bool isVisibilityToggleParam(const std::string& name) const {
    static const std::vector<std::string> names = {
      "tn_hcon_enable","tn_lcon_enable",
      "ptl_enable","ptm_enable",
      "brl_enable","brlp_enable",
      "hc_enable","hs_rgb_enable","hs_cmy_enable"
    };
    for (const auto& n : names) if (n == name) return true;
    return false;
  }

  bool almostEqual(float a, float b, float eps = 1e-6f) const {
    return std::fabs(a - b) <= eps;
  }

  // ===== Label Helpers: display name composition for clean/modified states =====
  std::string lookPresetDisplayName(int lookPresetIndex) const {
    if (!isUserLookPresetIndex(lookPresetIndex)) {
      return currentPresetName(lookPresetIndex);
    }
    int slot = -1;
    if (!userLookIndexFromPresetIndex(lookPresetIndex, &slot)) return std::string("Unknown User Look");
    std::lock_guard<std::mutex> lock(userPresetMutex());
    ensureUserPresetStoreLoadedLocked();
    const auto& s = userPresetStore().lookPresets[static_cast<size_t>(slot)];
    if (!s.name.empty()) return s.name;
    return std::string("User Look");
  }

  std::string presetLabelCleanForLook(int lookPresetIndex) const {
    return lookPresetDisplayName(lookPresetIndex) + " | " + buildLabelText();
  }

  std::string presetLabelCustomForLook(int lookPresetIndex) const {
    return std::string("Custom (") + lookPresetDisplayName(lookPresetIndex) + ") | " + buildLabelText();
  }

  // ===== Snapshot Capture: current UI values -> preset structs =====
  TonescalePresetValues captureCurrentTonescaleValues(double time) const {
    TonescalePresetValues t{};
    t.tn_con = getDouble("tn_con", time, 1.66f);
    t.tn_sh = getDouble("tn_sh", time, 0.5f);
    t.tn_toe = getDouble("tn_toe", time, 0.003f);
    t.tn_off = getDouble("tn_off", time, 0.005f);
    t.tn_hcon_enable = getBool("tn_hcon_enable", time, 0);
    t.tn_hcon = getDouble("tn_hcon", time, 0.0f);
    t.tn_hcon_pv = getDouble("tn_hcon_pv", time, 1.0f);
    t.tn_hcon_st = getDouble("tn_hcon_st", time, 4.0f);
    t.tn_lcon_enable = getBool("tn_lcon_enable", time, 0);
    t.tn_lcon = getDouble("tn_lcon", time, 0.0f);
    t.tn_lcon_w = getDouble("tn_lcon_w", time, 0.5f);
    return t;
  }

  LookPresetValues captureCurrentLookValues(double time) const {
    LookPresetValues v{};
    v.tn_con = getDouble("tn_con", time, 1.66f);
    v.tn_sh = getDouble("tn_sh", time, 0.5f);
    v.tn_toe = getDouble("tn_toe", time, 0.003f);
    v.tn_off = getDouble("tn_off", time, 0.005f);
    v.tn_hcon_enable = getBool("tn_hcon_enable", time, 0);
    v.tn_hcon = getDouble("tn_hcon", time, 0.0f);
    v.tn_hcon_pv = getDouble("tn_hcon_pv", time, 1.0f);
    v.tn_hcon_st = getDouble("tn_hcon_st", time, 4.0f);
    v.tn_lcon_enable = getBool("tn_lcon_enable", time, 0);
    v.tn_lcon = getDouble("tn_lcon", time, 0.0f);
    v.tn_lcon_w = getDouble("tn_lcon_w", time, 0.5f);
    v.cwp = getInt("cwp", time, 2);
    v.cwp_lm = getDouble("cwp_lm", time, 0.25f);
    v.rs_sa = getDouble("rs_sa", time, 0.35f);
    v.rs_rw = getDouble("rs_rw", time, 0.25f);
    v.rs_bw = getDouble("rs_bw", time, 0.55f);
    v.pt_enable = getBool("pt_enable", time, 1);
    v.pt_lml = getDouble("pt_lml", time, 0.25f);
    v.pt_lml_r = getDouble("pt_lml_r", time, 0.5f);
    v.pt_lml_g = getDouble("pt_lml_g", time, 0.0f);
    v.pt_lml_b = getDouble("pt_lml_b", time, 0.1f);
    v.pt_lmh = getDouble("pt_lmh", time, 0.25f);
    v.pt_lmh_r = getDouble("pt_lmh_r", time, 0.5f);
    v.pt_lmh_b = getDouble("pt_lmh_b", time, 0.0f);
    v.ptl_enable = getBool("ptl_enable", time, 1);
    v.ptl_c = getDouble("ptl_c", time, 0.06f);
    v.ptl_m = getDouble("ptl_m", time, 0.08f);
    v.ptl_y = getDouble("ptl_y", time, 0.06f);
    v.ptm_enable = getBool("ptm_enable", time, 1);
    v.ptm_low = getDouble("ptm_low", time, 0.4f);
    v.ptm_low_rng = getDouble("ptm_low_rng", time, 0.25f);
    v.ptm_low_st = getDouble("ptm_low_st", time, 0.5f);
    v.ptm_high = getDouble("ptm_high", time, -0.8f);
    v.ptm_high_rng = getDouble("ptm_high_rng", time, 0.35f);
    v.ptm_high_st = getDouble("ptm_high_st", time, 0.4f);
    v.brl_enable = getBool("brl_enable", time, 1);
    v.brl = getDouble("brl", time, 0.0f);
    v.brl_r = getDouble("brl_r", time, -2.5f);
    v.brl_g = getDouble("brl_g", time, -1.5f);
    v.brl_b = getDouble("brl_b", time, -1.5f);
    v.brl_rng = getDouble("brl_rng", time, 0.5f);
    v.brl_st = getDouble("brl_st", time, 0.35f);
    v.brlp_enable = getBool("brlp_enable", time, 1);
    v.brlp = getDouble("brlp", time, -0.5f);
    v.brlp_r = getDouble("brlp_r", time, -1.25f);
    v.brlp_g = getDouble("brlp_g", time, -1.25f);
    v.brlp_b = getDouble("brlp_b", time, -0.25f);
    v.hc_enable = getBool("hc_enable", time, 1);
    v.hc_r = getDouble("hc_r", time, 1.0f);
    v.hc_r_rng = getDouble("hc_r_rng", time, 0.3f);
    v.hs_rgb_enable = getBool("hs_rgb_enable", time, 1);
    v.hs_r = getDouble("hs_r", time, 0.6f);
    v.hs_r_rng = getDouble("hs_r_rng", time, 0.6f);
    v.hs_g = getDouble("hs_g", time, 0.35f);
    v.hs_g_rng = getDouble("hs_g_rng", time, 1.0f);
    v.hs_b = getDouble("hs_b", time, 0.66f);
    v.hs_b_rng = getDouble("hs_b_rng", time, 1.0f);
    v.hs_cmy_enable = getBool("hs_cmy_enable", time, 1);
    v.hs_c = getDouble("hs_c", time, 0.25f);
    v.hs_c_rng = getDouble("hs_c_rng", time, 1.0f);
    v.hs_m = getDouble("hs_m", time, 0.0f);
    v.hs_m_rng = getDouble("hs_m_rng", time, 1.0f);
    v.hs_y = getDouble("hs_y", time, 0.0f);
    v.hs_y_rng = getDouble("hs_y_rng", time, 1.0f);
    return v;
  }

  // ===== Preset Baseline Resolver =====
  // Computes the expected "clean" state for current look/tonescale/display selector choices.
  bool buildPresetBaseline(double time, OpenDRTParams* expected) const {
    if (expected == nullptr) return false;
    const int look = getChoice("lookPreset", time, 0);
    const int tsPreset = getChoice("tonescalePreset", time, 0);
    const int displayPreset = getChoice("displayEncodingPreset", time, 0);
    const int cwpPreset = getChoice("creativeWhitePreset", time, 0);
    OpenDRTParams out{};
    // Step 1: Start from look baseline (built-in or user look payload).
    if (isUserLookPresetIndex(look)) {
      int userIdx = -1;
      if (!userLookIndexFromPresetIndex(look, &userIdx)) return false;
      std::lock_guard<std::mutex> lock(userPresetMutex());
      ensureUserPresetStoreLoadedLocked();
      if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return false;
      const auto& s = userPresetStore().lookPresets[static_cast<size_t>(userIdx)];
      applyLookValuesToResolved(out, s.values);
    } else {
      applyLookPresetToResolved(out, look);
    }

    // Step 2: Apply tonescale policy:
    // - user tonescale preset wins
    // - selector==0 means "inherit tonescale from current look"
    // - otherwise use selected built-in tonescale preset
    if (isUserTonescalePresetIndex(tsPreset)) {
      int userIdx = -1;
      if (!userTonescaleIndexFromPresetIndex(tsPreset, &userIdx)) return false;
      std::lock_guard<std::mutex> lock(userPresetMutex());
      ensureUserPresetStoreLoadedLocked();
      if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().tonescalePresets.size())) return false;
      const auto& s = userPresetStore().tonescalePresets[static_cast<size_t>(userIdx)];
      applyTonescaleValuesToResolved(out, s.values);
    } else if (tsPreset == 0) {
      const TonescalePresetValues fromLook = selectedLookBaseTonescale(time);
      applyTonescaleValuesToResolved(out, fromLook);
    } else {
      applyTonescalePresetToResolved(out, tsPreset);
    }

    // Step 3: Apply display preset defaults.
    applyDisplayEncodingPreset(out, displayPreset);
    out.clamp = 1;
    // Step 4: Creative-white selector can override look baseline cwp.
    if (cwpPreset > 0) out.cwp = cwpPreset - 1;

    *expected = out;
    return true;
  }

  // ===== Category Reset Writers: apply selected baseline by section =====
  void applyTonescaleFromBaseline(const OpenDRTParams& p) {
    setDouble("tn_con", p.tn_con);
    setDouble("tn_sh", p.tn_sh);
    setDouble("tn_toe", p.tn_toe);
    setDouble("tn_off", p.tn_off);
    setBool("tn_hcon_enable", p.tn_hcon_enable != 0);
    setDouble("tn_hcon", p.tn_hcon);
    setDouble("tn_hcon_pv", p.tn_hcon_pv);
    setDouble("tn_hcon_st", p.tn_hcon_st);
    setBool("tn_lcon_enable", p.tn_lcon_enable != 0);
    setDouble("tn_lcon", p.tn_lcon);
    setDouble("tn_lcon_w", p.tn_lcon_w);
  }

  void applyRenderSpaceFromBaseline(const OpenDRTParams& p) {
    setDouble("rs_sa", p.rs_sa);
    setDouble("rs_rw", p.rs_rw);
    setDouble("rs_bw", p.rs_bw);
  }

  void applyMidPurityFromBaseline(const OpenDRTParams& p) {
    setBool("ptm_enable", p.ptm_enable != 0);
    setDouble("ptm_low", p.ptm_low);
    setDouble("ptm_low_rng", p.ptm_low_rng);
    setDouble("ptm_low_st", p.ptm_low_st);
    setDouble("ptm_high", p.ptm_high);
    setDouble("ptm_high_rng", p.ptm_high_rng);
    setDouble("ptm_high_st", p.ptm_high_st);
  }

  void applyPurityCompressionFromBaseline(const OpenDRTParams& p) {
    setBool("pt_enable", p.pt_enable != 0);
    setDouble("pt_lml", p.pt_lml);
    setDouble("pt_lml_r", p.pt_lml_r);
    setDouble("pt_lml_g", p.pt_lml_g);
    setDouble("pt_lml_b", p.pt_lml_b);
    setDouble("pt_lmh", p.pt_lmh);
    setDouble("pt_lmh_r", p.pt_lmh_r);
    setDouble("pt_lmh_b", p.pt_lmh_b);
    setBool("ptl_enable", p.ptl_enable != 0);
    setDouble("ptl_c", p.ptl_c);
    setDouble("ptl_m", p.ptl_m);
    setDouble("ptl_y", p.ptl_y);
  }

  void applyBrillianceFromBaseline(const OpenDRTParams& p) {
    setBool("brl_enable", p.brl_enable != 0);
    setDouble("brl", p.brl);
    setDouble("brl_r", p.brl_r);
    setDouble("brl_g", p.brl_g);
    setDouble("brl_b", p.brl_b);
    setDouble("brl_rng", p.brl_rng);
    setDouble("brl_st", p.brl_st);
    setBool("brlp_enable", p.brlp_enable != 0);
    setDouble("brlp", p.brlp);
    setDouble("brlp_r", p.brlp_r);
    setDouble("brlp_g", p.brlp_g);
    setDouble("brlp_b", p.brlp_b);
  }

  void applyHueFromBaseline(const OpenDRTParams& p) {
    setBool("hc_enable", p.hc_enable != 0);
    setDouble("hc_r", p.hc_r);
    setDouble("hc_r_rng", p.hc_r_rng);
    setBool("hs_rgb_enable", p.hs_rgb_enable != 0);
    setDouble("hs_r", p.hs_r);
    setDouble("hs_r_rng", p.hs_r_rng);
    setDouble("hs_g", p.hs_g);
    setDouble("hs_g_rng", p.hs_g_rng);
    setDouble("hs_b", p.hs_b);
    setDouble("hs_b_rng", p.hs_b_rng);
    setBool("hs_cmy_enable", p.hs_cmy_enable != 0);
    setDouble("hs_c", p.hs_c);
    setDouble("hs_c_rng", p.hs_c_rng);
    setDouble("hs_m", p.hs_m);
    setDouble("hs_m_rng", p.hs_m_rng);
    setDouble("hs_y", p.hs_y);
    setDouble("hs_y_rng", p.hs_y_rng);
  }

  // ===== Dirty-State Evaluation: compare live params against computed baseline =====
  bool isCurrentEqualToPresetBaseline(double time, bool* tonescaleCleanOut = nullptr) const {
    OpenDRTParams expected{};
    if (!buildPresetBaseline(time, &expected)) return false;

    // tonescaleClean is split out so we can mark tonescale menu "(Modified)" independently
    // from overall look modified state.
    const bool tonescaleClean =
      almostEqual(getDouble("tn_con", time, expected.tn_con), expected.tn_con) &&
      almostEqual(getDouble("tn_sh", time, expected.tn_sh), expected.tn_sh) &&
      almostEqual(getDouble("tn_toe", time, expected.tn_toe), expected.tn_toe) &&
      almostEqual(getDouble("tn_off", time, expected.tn_off), expected.tn_off) &&
      (getBool("tn_hcon_enable", time, expected.tn_hcon_enable) == expected.tn_hcon_enable) &&
      almostEqual(getDouble("tn_hcon", time, expected.tn_hcon), expected.tn_hcon) &&
      almostEqual(getDouble("tn_hcon_pv", time, expected.tn_hcon_pv), expected.tn_hcon_pv) &&
      almostEqual(getDouble("tn_hcon_st", time, expected.tn_hcon_st), expected.tn_hcon_st) &&
      (getBool("tn_lcon_enable", time, expected.tn_lcon_enable) == expected.tn_lcon_enable) &&
      almostEqual(getDouble("tn_lcon", time, expected.tn_lcon), expected.tn_lcon) &&
      almostEqual(getDouble("tn_lcon_w", time, expected.tn_lcon_w), expected.tn_lcon_w);

    if (tonescaleCleanOut) *tonescaleCleanOut = tonescaleClean;

    // Overall "clean" includes all preset-backed advanced controls + display settings + cwp/cwp_lm.
    const bool clean =
      tonescaleClean &&
      almostEqual(getDouble("rs_sa", time, expected.rs_sa), expected.rs_sa) &&
      almostEqual(getDouble("rs_rw", time, expected.rs_rw), expected.rs_rw) &&
      almostEqual(getDouble("rs_bw", time, expected.rs_bw), expected.rs_bw) &&
      (getBool("pt_enable", time, expected.pt_enable) == expected.pt_enable) &&
      almostEqual(getDouble("pt_lml", time, expected.pt_lml), expected.pt_lml) &&
      almostEqual(getDouble("pt_lml_r", time, expected.pt_lml_r), expected.pt_lml_r) &&
      almostEqual(getDouble("pt_lml_g", time, expected.pt_lml_g), expected.pt_lml_g) &&
      almostEqual(getDouble("pt_lml_b", time, expected.pt_lml_b), expected.pt_lml_b) &&
      almostEqual(getDouble("pt_lmh", time, expected.pt_lmh), expected.pt_lmh) &&
      almostEqual(getDouble("pt_lmh_r", time, expected.pt_lmh_r), expected.pt_lmh_r) &&
      almostEqual(getDouble("pt_lmh_b", time, expected.pt_lmh_b), expected.pt_lmh_b) &&
      (getBool("ptl_enable", time, expected.ptl_enable) == expected.ptl_enable) &&
      almostEqual(getDouble("ptl_c", time, expected.ptl_c), expected.ptl_c) &&
      almostEqual(getDouble("ptl_m", time, expected.ptl_m), expected.ptl_m) &&
      almostEqual(getDouble("ptl_y", time, expected.ptl_y), expected.ptl_y) &&
      (getBool("ptm_enable", time, expected.ptm_enable) == expected.ptm_enable) &&
      almostEqual(getDouble("ptm_low", time, expected.ptm_low), expected.ptm_low) &&
      almostEqual(getDouble("ptm_low_rng", time, expected.ptm_low_rng), expected.ptm_low_rng) &&
      almostEqual(getDouble("ptm_low_st", time, expected.ptm_low_st), expected.ptm_low_st) &&
      almostEqual(getDouble("ptm_high", time, expected.ptm_high), expected.ptm_high) &&
      almostEqual(getDouble("ptm_high_rng", time, expected.ptm_high_rng), expected.ptm_high_rng) &&
      almostEqual(getDouble("ptm_high_st", time, expected.ptm_high_st), expected.ptm_high_st) &&
      (getBool("brl_enable", time, expected.brl_enable) == expected.brl_enable) &&
      almostEqual(getDouble("brl", time, expected.brl), expected.brl) &&
      almostEqual(getDouble("brl_r", time, expected.brl_r), expected.brl_r) &&
      almostEqual(getDouble("brl_g", time, expected.brl_g), expected.brl_g) &&
      almostEqual(getDouble("brl_b", time, expected.brl_b), expected.brl_b) &&
      almostEqual(getDouble("brl_rng", time, expected.brl_rng), expected.brl_rng) &&
      almostEqual(getDouble("brl_st", time, expected.brl_st), expected.brl_st) &&
      (getBool("brlp_enable", time, expected.brlp_enable) == expected.brlp_enable) &&
      almostEqual(getDouble("brlp", time, expected.brlp), expected.brlp) &&
      almostEqual(getDouble("brlp_r", time, expected.brlp_r), expected.brlp_r) &&
      almostEqual(getDouble("brlp_g", time, expected.brlp_g), expected.brlp_g) &&
      almostEqual(getDouble("brlp_b", time, expected.brlp_b), expected.brlp_b) &&
      (getBool("hc_enable", time, expected.hc_enable) == expected.hc_enable) &&
      almostEqual(getDouble("hc_r", time, expected.hc_r), expected.hc_r) &&
      almostEqual(getDouble("hc_r_rng", time, expected.hc_r_rng), expected.hc_r_rng) &&
      (getBool("hs_rgb_enable", time, expected.hs_rgb_enable) == expected.hs_rgb_enable) &&
      almostEqual(getDouble("hs_r", time, expected.hs_r), expected.hs_r) &&
      almostEqual(getDouble("hs_r_rng", time, expected.hs_r_rng), expected.hs_r_rng) &&
      almostEqual(getDouble("hs_g", time, expected.hs_g), expected.hs_g) &&
      almostEqual(getDouble("hs_g_rng", time, expected.hs_g_rng), expected.hs_g_rng) &&
      almostEqual(getDouble("hs_b", time, expected.hs_b), expected.hs_b) &&
      almostEqual(getDouble("hs_b_rng", time, expected.hs_b_rng), expected.hs_b_rng) &&
      (getBool("hs_cmy_enable", time, expected.hs_cmy_enable) == expected.hs_cmy_enable) &&
      almostEqual(getDouble("hs_c", time, expected.hs_c), expected.hs_c) &&
      almostEqual(getDouble("hs_c_rng", time, expected.hs_c_rng), expected.hs_c_rng) &&
      almostEqual(getDouble("hs_m", time, expected.hs_m), expected.hs_m) &&
      almostEqual(getDouble("hs_m_rng", time, expected.hs_m_rng), expected.hs_m_rng) &&
      almostEqual(getDouble("hs_y", time, expected.hs_y), expected.hs_y) &&
      almostEqual(getDouble("hs_y_rng", time, expected.hs_y_rng), expected.hs_y_rng) &&
      (getInt("cwp", time, expected.cwp) == expected.cwp) &&
      almostEqual(getDouble("cwp_lm", time, expected.cwp_lm), expected.cwp_lm) &&
      (getBool("clamp", time, expected.clamp) == expected.clamp) &&
      (getChoice("tn_su", time, expected.tn_su) == expected.tn_su) &&
      (getChoice("display_gamut", time, expected.display_gamut) == expected.display_gamut) &&
      (getChoice("eotf", time, expected.eotf) == expected.eotf);

    return clean;
  }

  void updatePresetStateFromCurrent(double time) {
    bool tonescaleClean = true;
    const bool clean = isCurrentEqualToPresetBaseline(time, &tonescaleClean);
    // presetState drives UI readout and Discard availability.
    setInt("presetState", clean ? 0 : 1);
    if (auto* p = fetchPushButtonParam("discardPresetChanges")) p->setEnabled(!clean);
    // Menu label mutation is separate so users can see "(Modified)" directly in look/tonescale lists.
    applyPresetMenuModifiedLabels(time, !clean, !tonescaleClean);
  }

  // ===== Typed OFX Param Accessors =====
  int getChoice(const char* name, double t, int def) const {
    if (auto* p = fetchChoiceParam(name)) {
      int v = def;
      p->getValueAtTime(t, v);
      return v;
    }
    return def;
  }
  int getInt(const char* name, double t, int def) const {
    if (auto* p = fetchIntParam(name)) return p->getValueAtTime(t);
    return def;
  }
  int getBool(const char* name, double t, int def) const {
    if (auto* p = fetchBooleanParam(name)) return p->getValueAtTime(t) ? 1 : 0;
    return def;
  }
  float getDouble(const char* name, double t, float def) const {
    if (auto* p = fetchDoubleParam(name)) return static_cast<float>(p->getValueAtTime(t));
    return def;
  }
  std::string getString(const char* name, const std::string& def) const {
    if (auto* p = fetchStringParam(name)) {
      std::string v = def;
      p->getValue(v);
      return v;
    }
    return def;
  }
  void setBool(const char* name, bool v) {
    if (auto* p = fetchBooleanParam(name)) p->setValue(v);
  }
  void setDouble(const char* name, double v) {
    if (auto* p = fetchDoubleParam(name)) p->setValue(v);
  }
  void setChoice(const char* name, int v) {
    if (auto* p = fetchChoiceParam(name)) p->setValue(v);
  }

  // ===== Look-Derived Defaults: base whitepoint and tonescale from selected look =====
  int selectedLookBaseCwp(double t) const {
    const int lookIdx = getChoice("lookPreset", t, 0);
    if (isUserLookPresetIndex(lookIdx)) {
      int userIdx = -1;
      if (!userLookIndexFromPresetIndex(lookIdx, &userIdx)) return 2;
      std::lock_guard<std::mutex> lock(userPresetMutex());
      ensureUserPresetStoreLoadedLocked();
      if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return 2;
      return userPresetStore().lookPresets[static_cast<size_t>(userIdx)].values.cwp;
    }
    if (lookIdx < 0 || lookIdx >= static_cast<int>(kLookPresets.size())) return 2;
    return kLookPresets[static_cast<size_t>(lookIdx)].cwp;
  }

  TonescalePresetValues selectedLookBaseTonescale(double t) const {
    const int lookIdx = getChoice("lookPreset", t, 0);
    TonescalePresetValues out{};
    if (isUserLookPresetIndex(lookIdx)) {
      int userIdx = -1;
      if (!userLookIndexFromPresetIndex(lookIdx, &userIdx)) return captureCurrentTonescaleValues(t);
      std::lock_guard<std::mutex> lock(userPresetMutex());
      ensureUserPresetStoreLoadedLocked();
      if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return captureCurrentTonescaleValues(t);
      const auto& lv = userPresetStore().lookPresets[static_cast<size_t>(userIdx)].values;
      out.tn_con = lv.tn_con;
      out.tn_sh = lv.tn_sh;
      out.tn_toe = lv.tn_toe;
      out.tn_off = lv.tn_off;
      out.tn_hcon_enable = lv.tn_hcon_enable;
      out.tn_hcon = lv.tn_hcon;
      out.tn_hcon_pv = lv.tn_hcon_pv;
      out.tn_hcon_st = lv.tn_hcon_st;
      out.tn_lcon_enable = lv.tn_lcon_enable;
      out.tn_lcon = lv.tn_lcon;
      out.tn_lcon_w = lv.tn_lcon_w;
      return out;
    }
    const int idx = (lookIdx < 0 || lookIdx >= static_cast<int>(kLookPresets.size())) ? 0 : lookIdx;
    const auto& lv = kLookPresets[static_cast<size_t>(idx)];
    out.tn_con = lv.tn_con;
    out.tn_sh = lv.tn_sh;
    out.tn_toe = lv.tn_toe;
    out.tn_off = lv.tn_off;
    out.tn_hcon_enable = lv.tn_hcon_enable;
    out.tn_hcon = lv.tn_hcon;
    out.tn_hcon_pv = lv.tn_hcon_pv;
    out.tn_hcon_st = lv.tn_hcon_st;
    out.tn_lcon_enable = lv.tn_lcon_enable;
    out.tn_lcon = lv.tn_lcon;
    out.tn_lcon_w = lv.tn_lcon_w;
    return out;
  }

  // ===== Menu Label Mutation: attach/remove "(Modified)" without rebuilding options =====
  std::string lookBaseMenuName(int idx) const {
    if (idx >= 0 && idx < kBuiltInLookPresetCount) return std::string(kLookPresetNames[static_cast<size_t>(idx)]);
    if (!isUserLookPresetIndex(idx)) return std::string();
    int userIdx = -1;
    if (!userLookIndexFromPresetIndex(idx, &userIdx)) return std::string();
    std::lock_guard<std::mutex> lock(userPresetMutex());
    ensureUserPresetStoreLoadedLocked();
    if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().lookPresets.size())) return std::string();
    return userPresetStore().lookPresets[static_cast<size_t>(userIdx)].name;
  }

  std::string tonescaleBaseMenuName(int idx) const {
    if (idx >= 0 && idx < kBuiltInTonescalePresetCount) return std::string(kTonescalePresetNames[static_cast<size_t>(idx)]);
    if (!isUserTonescalePresetIndex(idx)) return std::string();
    int userIdx = -1;
    if (!userTonescaleIndexFromPresetIndex(idx, &userIdx)) return std::string();
    std::lock_guard<std::mutex> lock(userPresetMutex());
    ensureUserPresetStoreLoadedLocked();
    if (userIdx < 0 || userIdx >= static_cast<int>(userPresetStore().tonescalePresets.size())) return std::string();
    return userPresetStore().tonescalePresets[static_cast<size_t>(userIdx)].name;
  }

  void applyPresetMenuModifiedLabels(double t, bool lookModified, bool tonescaleModified) {
    const int lookIdx = getChoice("lookPreset", t, 0);
    const int toneIdx = getChoice("tonescalePreset", t, 0);
    if (menuLabelCacheInit_ &&
        lookIdx == menuLabelLookIdx_ &&
        toneIdx == menuLabelToneIdx_ &&
        lookModified == menuLabelLookModified_ &&
        tonescaleModified == menuLabelToneModified_) {
      // Fast path: avoid repeated setOption churn when nothing changed.
      return;
    }
    auto* lookParam = fetchChoiceParam("lookPreset");
    auto* toneParam = fetchChoiceParam("tonescalePreset");

    if (lookParam && menuLabelCacheInit_ && menuLabelLookIdx_ >= 0) {
      // Restore previous option text before applying new modified suffix.
      const std::string basePrev = lookBaseMenuName(menuLabelLookIdx_);
      if (!basePrev.empty()) lookParam->setOption(menuLabelLookIdx_, basePrev);
    }
    if (toneParam && menuLabelCacheInit_ && menuLabelToneIdx_ >= 0) {
      const std::string basePrev = tonescaleBaseMenuName(menuLabelToneIdx_);
      if (!basePrev.empty()) toneParam->setOption(menuLabelToneIdx_, basePrev);
    }

    if (lookParam) {
      const std::string base = lookBaseMenuName(lookIdx);
      if (!base.empty() && lookModified) lookParam->setOption(lookIdx, base + " (Modified)");
    }
    if (toneParam) {
      const std::string base = tonescaleBaseMenuName(toneIdx);
      if (!base.empty() && tonescaleModified) toneParam->setOption(toneIdx, base + " (Modified)");
    }

    menuLabelLookIdx_ = lookIdx;
    menuLabelToneIdx_ = toneIdx;
    menuLabelLookModified_ = lookModified;
    menuLabelToneModified_ = tonescaleModified;
    menuLabelCacheInit_ = true;
  }

  // ===== Readonly Labels: effective whitepoint/surround UI fields =====
  void updateReadonlyDisplayLabels(double t) {
    const int lookIdx = getChoice("lookPreset", t, 0);
    const int cwpPreset = getChoice("creativeWhitePreset", t, 0);
    const int tnSu = getChoice("tn_su", t, 1);
    const int cwp = (cwpPreset <= 0) ? selectedLookBaseCwp(t) : (cwpPreset - 1);
    std::string baseWp = std::string(whitepointNameFromCwp(cwp));
    if (isUserLookPresetIndex(lookIdx) && cwpPreset <= 0) {
      baseWp += " (User)";
    }
    setString("baseWhitepointLabel", baseWp);
    setString("surroundLabel", surroundNameFromIndex(tnSu));
  }

  // ===== Menu Rebuild: reconstruct look/tonescale choice options from store =====
  void rebuildLookPresetMenuOptions(int preferredIndex) {
    auto* p = fetchChoiceParam("lookPreset");
    if (!p) return;
    p->resetOptions();
    for (const char* n : kLookPresetNames) p->appendOption(n);
    int userCount = 0;
    {
      std::lock_guard<std::mutex> lock(userPresetMutex());
      ensureUserPresetStoreLoadedLocked();
      for (const auto& u : userPresetStore().lookPresets) p->appendOption(u.name);
      userCount = static_cast<int>(userPresetStore().lookPresets.size());
    }
    const int maxIndex = kBuiltInLookPresetCount + userCount - 1;
    const int clamped = preferredIndex < 0 ? 0 : (preferredIndex > maxIndex ? maxIndex : preferredIndex);
    p->setValue(clamped);
  }

  void rebuildTonescalePresetMenuOptions(int preferredIndex) {
    auto* p = fetchChoiceParam("tonescalePreset");
    if (!p) return;
    p->resetOptions();
    for (const char* n : kTonescalePresetNames) p->appendOption(n);
    int userCount = 0;
    {
      std::lock_guard<std::mutex> lock(userPresetMutex());
      ensureUserPresetStoreLoadedLocked();
      for (const auto& u : userPresetStore().tonescalePresets) p->appendOption(u.name);
      userCount = static_cast<int>(userPresetStore().tonescalePresets.size());
    }
    const int maxIndex = kBuiltInTonescalePresetCount + userCount - 1;
    const int clamped = preferredIndex < 0 ? 0 : (preferredIndex > maxIndex ? maxIndex : preferredIndex);
    p->setValue(clamped);
  }

  void rebuildAllPresetMenus(int preferredLookIndex, int preferredToneIndex) {
    // Reset cached "(Modified)" menu label state whenever options are rebuilt.
    menuLabelCacheInit_ = false;
    menuLabelLookIdx_ = -1;
    menuLabelToneIdx_ = -1;
    menuLabelLookModified_ = false;
    menuLabelToneModified_ = false;
    rebuildLookPresetMenuOptions(preferredLookIndex);
    rebuildTonescalePresetMenuOptions(preferredToneIndex);
  }

  // Source-of-truth menu refresh:
  // reload disk store, clamp selection indices, rebuild both menus, then refresh dependent UI state.
  // ===== Preset Menu Sync =====
  // Reload file -> rebuild look/tonescale menus -> refresh dependent preset labels/state.
  void syncPresetMenusFromDisk(double t, int preferredLookIndex, int preferredToneIndex) {
    int lookPreferred = preferredLookIndex;
    int tonePreferred = preferredToneIndex;
    {
      std::lock_guard<std::mutex> lock(userPresetMutex());
      // Refresh source of truth from disk first, then clamp requested selections to valid ranges.
      reloadUserPresetStoreFromDiskLocked();
      const int maxLook = kBuiltInLookPresetCount + static_cast<int>(userPresetStore().lookPresets.size()) - 1;
      const int maxTone = kBuiltInTonescalePresetCount + static_cast<int>(userPresetStore().tonescalePresets.size()) - 1;
      if (lookPreferred < 0 || lookPreferred > maxLook) lookPreferred = 0;
      if (tonePreferred < 0 || tonePreferred > maxTone) tonePreferred = 0;
    }
    rebuildAllPresetMenus(lookPreferred, tonePreferred);
    updatePresetManagerActionState(t);
    updatePresetStateFromCurrent(t);
    updateReadonlyDisplayLabels(t);
  }

  // ===== Preset Manager Button Enable Rules =====
  // Manager actions are enabled only when current look or tonescale points to a user preset.
  void updatePresetManagerActionState(double t) {
    const int lookIdx = getChoice("lookPreset", t, 0);
    const int toneIdx = getChoice("tonescalePreset", t, 0);
    const bool enable = isUserLookPresetIndex(lookIdx) || isUserTonescalePresetIndex(toneIdx);
    if (auto* p = fetchPushButtonParam("userPresetUpdateCurrent")) p->setEnabled(enable);
    if (auto* p = fetchPushButtonParam("userPresetDeleteCurrent")) p->setEnabled(enable);
    if (auto* p = fetchPushButtonParam("userPresetRenameCurrent")) p->setEnabled(enable);
  }

  void setParamVisible(const char* name, bool visible) {
    try {
      if (auto* p = fetchDoubleParam(name)) { p->setIsSecret(!visible); p->setEnabled(visible); return; }
      if (auto* p = fetchBooleanParam(name)) { p->setIsSecret(!visible); p->setEnabled(visible); return; }
      if (auto* p = fetchChoiceParam(name)) { p->setIsSecret(!visible); p->setEnabled(visible); return; }
      if (auto* p = fetchIntParam(name)) { p->setIsSecret(!visible); p->setEnabled(visible); return; }
      if (auto* p = fetchStringParam(name)) { p->setIsSecret(!visible); p->setEnabled(visible); return; }
    } catch (...) {
    }
  }

  // Advanced toggle visibility updater.
  // Uses a small cache to avoid calling setIsSecret/setEnabled unless a driving toggle changed.
  void updateToggleVisibility(double t) {
    const bool hcon = getBool("tn_hcon_enable", t, 0) != 0;
    const bool lcon = getBool("tn_lcon_enable", t, 0) != 0;
    const bool ptl = getBool("ptl_enable", t, 1) != 0;
    const bool ptm = getBool("ptm_enable", t, 1) != 0;
    const bool brl = getBool("brl_enable", t, 1) != 0;
    const bool brlp = getBool("brlp_enable", t, 1) != 0;
    const bool hc = getBool("hc_enable", t, 1) != 0;
    const bool hsRgb = getBool("hs_rgb_enable", t, 1) != 0;
    const bool hsCmy = getBool("hs_cmy_enable", t, 1) != 0;

    if (visibilityCacheInit_ &&
        hcon == vis_hcon_ &&
        lcon == vis_lcon_ &&
        ptl == vis_ptl_ &&
        ptm == vis_ptm_ &&
        brl == vis_brl_ &&
        brlp == vis_brlp_ &&
        hc == vis_hc_ &&
        hsRgb == vis_hsRgb_ &&
        hsCmy == vis_hsCmy_) {
      return;
    }

    const bool applyHcon = !visibilityCacheInit_ || hcon != vis_hcon_;
    const bool applyLcon = !visibilityCacheInit_ || lcon != vis_lcon_;
    const bool applyPtl = !visibilityCacheInit_ || ptl != vis_ptl_;
    const bool applyPtm = !visibilityCacheInit_ || ptm != vis_ptm_;
    const bool applyBrl = !visibilityCacheInit_ || brl != vis_brl_;
    const bool applyBrlp = !visibilityCacheInit_ || brlp != vis_brlp_;
    const bool applyHc = !visibilityCacheInit_ || hc != vis_hc_;
    const bool applyHsRgb = !visibilityCacheInit_ || hsRgb != vis_hsRgb_;
    const bool applyHsCmy = !visibilityCacheInit_ || hsCmy != vis_hsCmy_;

    if (applyHcon) {
      setParamVisible("tn_hcon", hcon);
      setParamVisible("tn_hcon_pv", hcon);
      setParamVisible("tn_hcon_st", hcon);
    }
    if (applyLcon) {
      setParamVisible("tn_lcon", lcon);
      setParamVisible("tn_lcon_w", lcon);
    }
    if (applyPtl) {
      setParamVisible("ptl_c", ptl);
      setParamVisible("ptl_m", ptl);
      setParamVisible("ptl_y", ptl);
    }
    if (applyPtm) {
      setParamVisible("ptm_low", ptm);
      setParamVisible("ptm_low_rng", ptm);
      setParamVisible("ptm_low_st", ptm);
      setParamVisible("ptm_high", ptm);
      setParamVisible("ptm_high_rng", ptm);
      setParamVisible("ptm_high_st", ptm);
    }
    if (applyBrl) {
      setParamVisible("brl", brl);
      setParamVisible("brl_r", brl);
      setParamVisible("brl_g", brl);
      setParamVisible("brl_b", brl);
      setParamVisible("brl_rng", brl);
      setParamVisible("brl_st", brl);
    }
    if (applyBrlp) {
      setParamVisible("brlp", brlp);
      setParamVisible("brlp_r", brlp);
      setParamVisible("brlp_g", brlp);
      setParamVisible("brlp_b", brlp);
    }
    if (applyHc) {
      setParamVisible("hc_r", hc);
      setParamVisible("hc_r_rng", hc);
    }
    if (applyHsRgb) {
      setParamVisible("hs_r", hsRgb);
      setParamVisible("hs_r_rng", hsRgb);
      setParamVisible("hs_g", hsRgb);
      setParamVisible("hs_g_rng", hsRgb);
      setParamVisible("hs_b", hsRgb);
      setParamVisible("hs_b_rng", hsRgb);
    }
    if (applyHsCmy) {
      setParamVisible("hs_c", hsCmy);
      setParamVisible("hs_c_rng", hsCmy);
      setParamVisible("hs_m", hsCmy);
      setParamVisible("hs_m_rng", hsCmy);
      setParamVisible("hs_y", hsCmy);
      setParamVisible("hs_y_rng", hsCmy);
    }

    vis_hcon_ = hcon;
    vis_lcon_ = lcon;
    vis_ptl_ = ptl;
    vis_ptm_ = ptm;
    vis_brl_ = brl;
    vis_brlp_ = brlp;
    vis_hc_ = hc;
    vis_hsRgb_ = hsRgb;
    vis_hsCmy_ = hsCmy;
    visibilityCacheInit_ = true;
  }
  void setInt(const char* name, int v) {
    if (auto* p = fetchIntParam(name)) p->setValue(v);
  }
  void setString(const char* name, const std::string& v) {
    if (auto* p = fetchStringParam(name)) p->setValue(v);
  }

  OpenDRTRawValues readRawValues(double time) const {
    OpenDRTRawValues r{};
    r.in_gamut = getChoice("in_gamut", time, 14);
    r.in_oetf = getChoice("in_oetf", time, 1);
    r.tn_Lp = getDouble("tn_Lp", time, 100.0f);
    r.tn_gb = getDouble("tn_gb", time, 0.13f);
    r.pt_hdr = getDouble("pt_hdr", time, 0.5f);
    r.tn_Lg = getDouble("tn_Lg", time, 10.0f);
    r.crv_enable = getBool("crv_enable", time, 0);
    r.lookPreset = getChoice("lookPreset", time, 0);
    r.tonescalePreset = getChoice("tonescalePreset", time, 0);
    r.creativeWhitePreset = getChoice("creativeWhitePreset", time, 0);
    r.cwp = getInt("cwp", time, 2);
    r.creativeWhiteLimit = getDouble("cwp_lm", time, 0.25f);
    r.displayEncodingPreset = getChoice("displayEncodingPreset", time, 0);

    r.tn_con = getDouble("tn_con", time, 1.66f);
    r.tn_sh = getDouble("tn_sh", time, 0.5f);
    r.tn_toe = getDouble("tn_toe", time, 0.003f);
    r.tn_off = getDouble("tn_off", time, 0.005f);
    r.tn_hcon_enable = getBool("tn_hcon_enable", time, 0);
    r.tn_hcon = getDouble("tn_hcon", time, 0.0f);
    r.tn_hcon_pv = getDouble("tn_hcon_pv", time, 1.0f);
    r.tn_hcon_st = getDouble("tn_hcon_st", time, 4.0f);
    r.tn_lcon_enable = getBool("tn_lcon_enable", time, 0);
    r.tn_lcon = getDouble("tn_lcon", time, 0.0f);
    r.tn_lcon_w = getDouble("tn_lcon_w", time, 0.5f);

    r.rs_sa = getDouble("rs_sa", time, 0.35f);
    r.rs_rw = getDouble("rs_rw", time, 0.25f);
    r.rs_bw = getDouble("rs_bw", time, 0.55f);

    r.pt_enable = getBool("pt_enable", time, 1);
    r.pt_lml = getDouble("pt_lml", time, 0.25f);
    r.pt_lml_r = getDouble("pt_lml_r", time, 0.5f);
    r.pt_lml_g = getDouble("pt_lml_g", time, 0.0f);
    r.pt_lml_b = getDouble("pt_lml_b", time, 0.1f);
    r.pt_lmh = getDouble("pt_lmh", time, 0.25f);
    r.pt_lmh_r = getDouble("pt_lmh_r", time, 0.5f);
    r.pt_lmh_b = getDouble("pt_lmh_b", time, 0.0f);
    r.ptl_enable = getBool("ptl_enable", time, 1);
    r.ptl_c = getDouble("ptl_c", time, 0.06f);
    r.ptl_m = getDouble("ptl_m", time, 0.08f);
    r.ptl_y = getDouble("ptl_y", time, 0.06f);
    r.ptm_enable = getBool("ptm_enable", time, 1);
    r.ptm_low = getDouble("ptm_low", time, 0.4f);
    r.ptm_low_rng = getDouble("ptm_low_rng", time, 0.25f);
    r.ptm_low_st = getDouble("ptm_low_st", time, 0.5f);
    r.ptm_high = getDouble("ptm_high", time, -0.8f);
    r.ptm_high_rng = getDouble("ptm_high_rng", time, 0.35f);
    r.ptm_high_st = getDouble("ptm_high_st", time, 0.4f);

    r.brl_enable = getBool("brl_enable", time, 1);
    r.brl = getDouble("brl", time, 0.0f);
    r.brl_r = getDouble("brl_r", time, -2.5f);
    r.brl_g = getDouble("brl_g", time, -1.5f);
    r.brl_b = getDouble("brl_b", time, -1.5f);
    r.brl_rng = getDouble("brl_rng", time, 0.5f);
    r.brl_st = getDouble("brl_st", time, 0.35f);
    r.brlp_enable = getBool("brlp_enable", time, 1);
    r.brlp = getDouble("brlp", time, -0.5f);
    r.brlp_r = getDouble("brlp_r", time, -1.25f);
    r.brlp_g = getDouble("brlp_g", time, -1.25f);
    r.brlp_b = getDouble("brlp_b", time, -0.25f);

    r.hc_enable = getBool("hc_enable", time, 1);
    r.hc_r = getDouble("hc_r", time, 1.0f);
    r.hc_r_rng = getDouble("hc_r_rng", time, 0.3f);
    r.hs_rgb_enable = getBool("hs_rgb_enable", time, 1);
    r.hs_r = getDouble("hs_r", time, 0.6f);
    r.hs_r_rng = getDouble("hs_r_rng", time, 0.6f);
    r.hs_g = getDouble("hs_g", time, 0.35f);
    r.hs_g_rng = getDouble("hs_g_rng", time, 1.0f);
    r.hs_b = getDouble("hs_b", time, 0.66f);
    r.hs_b_rng = getDouble("hs_b_rng", time, 1.0f);
    r.hs_cmy_enable = getBool("hs_cmy_enable", time, 1);
    r.hs_c = getDouble("hs_c", time, 0.25f);
    r.hs_c_rng = getDouble("hs_c_rng", time, 1.0f);
    r.hs_m = getDouble("hs_m", time, 0.0f);
    r.hs_m_rng = getDouble("hs_m_rng", time, 1.0f);
    r.hs_y = getDouble("hs_y", time, 0.0f);
    r.hs_y_rng = getDouble("hs_y_rng", time, 1.0f);

    r.clamp = getBool("clamp", time, 1);
    r.tn_su = getChoice("tn_su", time, 1);
    r.display_gamut = getChoice("display_gamut", time, 0);
    r.eotf = getChoice("eotf", time, 2);

    return r;
  }

  OFX::Clip* dstClip_ = nullptr;
  OFX::Clip* srcClip_ = nullptr;
  std::unique_ptr<OpenDRTProcessor> processor_;
  std::vector<float> srcPixels_;
  std::vector<float> dstPixels_;
#if defined(OFX_SUPPORTS_CUDARENDER)
  float* stageSrcPinned_ = nullptr;
  float* stageDstPinned_ = nullptr;
  size_t stagePinnedCapacityFloats_ = 0;
#endif
  bool suppressParamChanged_ = false;
  bool visibilityCacheInit_ = false;
  bool vis_hcon_ = false;
  bool vis_lcon_ = false;
  bool vis_ptl_ = false;
  bool vis_ptm_ = false;
  bool vis_brl_ = false;
  bool vis_brlp_ = false;
  bool vis_hc_ = false;
  bool vis_hsRgb_ = false;
  bool vis_hsCmy_ = false;
  bool menuLabelCacheInit_ = false;
  int menuLabelLookIdx_ = -1;
  int menuLabelToneIdx_ = -1;
  bool menuLabelLookModified_ = false;
  bool menuLabelToneModified_ = false;
};

class OpenDRTFactory : public OFX::PluginFactoryHelper<OpenDRTFactory> {
 public:
  OpenDRTFactory() : PluginFactoryHelper<OpenDRTFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}

  void load() override {}
  void unload() override {}

  // ===== Plugin Descriptor =====
  // Host capability advertisement and static metadata.
  void describe(OFX::ImageEffectDescriptor& d) override {
    static const std::string nameWithVersion = "ME_OpenDRT v1.1";
    d.setLabels(nameWithVersion.c_str(), nameWithVersion.c_str(), nameWithVersion.c_str());
    d.setPluginGrouping(kPluginGrouping);
    d.setPluginDescription(std::string(kPluginDescription) + " | " + buildLabelText());
    d.addSupportedContext(OFX::eContextFilter);
    d.addSupportedBitDepth(OFX::eBitDepthFloat);
    d.setSingleInstance(false);
    d.setSupportsTiles(false);
    d.setSupportsMultiResolution(false);
    d.setTemporalClipAccess(false);
    d.setSupportsOpenCLBuffersRender(false);
#if defined(OFX_SUPPORTS_CUDARENDER)
    const bool advertiseHostCuda = (selectedCudaRenderMode() == CudaRenderMode::HostPreferred);
    d.setSupportsCudaRender(advertiseHostCuda);
    d.setSupportsCudaStream(advertiseHostCuda);
#elif defined(__APPLE__)
    const bool advertiseHostMetal = (selectedMetalRenderMode() == MetalRenderMode::HostPreferred);
    d.setSupportsMetalRender(advertiseHostMetal);
    d.setSupportsCudaRender(false);
    d.setSupportsCudaStream(false);
#else
    d.setSupportsCudaRender(false);
    d.setSupportsCudaStream(false);
#endif
  }

  // ===== Parameter + UI Layout =====
  // Defines all OFX params, groups, pages, and parent/child wiring.
  void describeInContext(OFX::ImageEffectDescriptor& d, OFX::ContextEnum) override {
    OFX::ClipDescriptor* src = d.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);

    OFX::ClipDescriptor* dst = d.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    auto* pLook = d.definePageParam("Look");
    auto* pOverlay = d.definePageParam("Overlay");
    auto* pAdvanced = d.definePageParam("Advanced Look Control");
    auto* pUserPresets = d.definePageParam("User Preset Manager");
    auto* grpUserPresetsRoot = d.defineGroupParam("grp_user_presets_root");
    grpUserPresetsRoot->setLabel("User Preset Manager");
    grpUserPresetsRoot->setOpen(false);

    auto addChoice = [&d](const char* name, const char* label, int def, const std::vector<const char*>& opts) {
      auto* p = d.defineChoiceParam(name);
      p->setLabel(label);
      for (const char* o : opts) p->appendOption(o);
      p->setDefault(def);
      if (const char* hint = tooltipForParam(name)) p->setHint(hint);
      return p;
    };
    auto addDouble = [&d](const char* name, const char* label, double def, double mn, double mx) {
      auto* p = d.defineDoubleParam(name);
      p->setLabel(label);
      p->setDefault(def);
      p->setRange(mn, mx);
      p->setDisplayRange(mn, mx);
      if (const char* hint = tooltipForParam(name)) p->setHint(hint);
      return p;
    };

    auto* inGamut = addChoice("in_gamut", "Input Gamut", 14, {"XYZ","ACES 2065-1","ACEScg","P3D65","Rec.2020","Rec.709","Arri Wide Gamut 3","Arri Wide Gamut 4","Red Wide Gamut RGB","Sony SGamut3","Sony SGamut3Cine","Panasonic V-Gamut","Filmlight E-Gamut","Filmlight E-Gamut2","DaVinci Wide Gamut"});
    auto* inOetf = addChoice("in_oetf", "Input Transfer Function", 1, {"Linear","DaVinci Intermediate","Filmlight T-Log","ACEScct","Arri LogC3","Arri LogC4","RedLog3G10","Panasonic V-Log","Sony S-Log3","Fuji F-Log2"});

    auto* dep = addChoice("displayEncodingPreset", "Display Encoding Preset", 0, {"Rec.1886 - 2.4 Power / Rec.709","sRGB Display - 2.2 Power / Rec.709","Display P3 - 2.2 Power / P3-D65","DCI - 2.6 Power / P3-D60","DCI - 2.6 Power / P3-DCI","DCI - 2.6 Power / XYZ","Rec.2100 - PQ / Rec.2020","Rec.2100 - HLG / Rec.2020","Dolby - PQ / P3-D65"});
    auto* lookPreset = addChoice("lookPreset", "Look Preset", 0, {"Standard","Arriba","Sylvan","Colorful","Aery","Dystopic","Umbra","Base"});
    for (const auto& n : visibleUserLookNames()) lookPreset->appendOption(n);
    auto* presetState = d.defineIntParam("presetState"); presetState->setIsSecret(true); presetState->setDefault(0);
    auto* cwpHidden = d.defineIntParam("cwp"); cwpHidden->setIsSecret(true); cwpHidden->setDefault(2);
    auto* activeUserLookSlot = d.defineIntParam("activeUserLookSlot"); activeUserLookSlot->setIsSecret(true); activeUserLookSlot->setDefault(-1);
    auto* activeUserToneSlot = d.defineIntParam("activeUserToneSlot"); activeUserToneSlot->setIsSecret(true); activeUserToneSlot->setDefault(-1);
    auto* tonescalePreset = addChoice("tonescalePreset", "Tonescale Preset", 0, {"USE LOOK PRESET","Low Contrast","Medium Contrast","High Contrast","Arriba Tonescale","Sylvan Tonescale","Colorful Tonescale","Aery Tonescale","Dystopic Tonescale","Umbra Tonescale","ACES-1.x","ACES-2.0","Marvelous Tonescape","DaGrinchi ToneGroan"});
    for (const auto& n : visibleUserTonescaleNames()) tonescalePreset->appendOption(n);
    auto* cwpPreset = addChoice("creativeWhitePreset", "Creative White", 0, {"USE LOOK PRESET","D93","D75","D65","D60","D55","D50"});
    auto* cwpLm = addDouble("cwp_lm", "Creative White Limit", 0.25, 0.0, 1.0);
    auto* baseWpLabel = d.defineStringParam("baseWhitepointLabel");
    baseWpLabel->setLabel("Effective Whitepoint");
    baseWpLabel->setDefault("D65");
    baseWpLabel->setEnabled(false);
    auto* surroundLabel = d.defineStringParam("surroundLabel");
    surroundLabel->setLabel("Effective Surround");
    surroundLabel->setDefault("Dim");
    surroundLabel->setEnabled(false);
    auto* grpLookBasic = d.defineGroupParam("grp_look_basic");
    grpLookBasic->setLabel("Basic");
    grpLookBasic->setOpen(true);
    inGamut->setParent(*grpLookBasic);
    inOetf->setParent(*grpLookBasic);
    dep->setParent(*grpLookBasic);
    lookPreset->setParent(*grpLookBasic);
    presetState->setParent(*grpLookBasic);
    cwpHidden->setParent(*grpLookBasic);
    activeUserLookSlot->setParent(*grpLookBasic);
    activeUserToneSlot->setParent(*grpLookBasic);
    tonescalePreset->setParent(*grpLookBasic);
    cwpPreset->setParent(*grpLookBasic);
    cwpLm->setParent(*grpLookBasic);
    baseWpLabel->setParent(*grpLookBasic);
    surroundLabel->setParent(*grpLookBasic);
    auto* discardPresetChanges = d.definePushButtonParam("discardPresetChanges");
    discardPresetChanges->setLabel("Discard Changes");
    discardPresetChanges->setEnabled(false);
    discardPresetChanges->setParent(*grpLookBasic);
    pLook->addChild(*grpLookBasic); pLook->addChild(*inGamut); pLook->addChild(*inOetf); pLook->addChild(*dep); pLook->addChild(*lookPreset); pLook->addChild(*presetState); pLook->addChild(*cwpHidden); pLook->addChild(*activeUserLookSlot); pLook->addChild(*activeUserToneSlot); pLook->addChild(*tonescalePreset); pLook->addChild(*cwpPreset); pLook->addChild(*cwpLm); pLook->addChild(*baseWpLabel); pLook->addChild(*surroundLabel); pLook->addChild(*discardPresetChanges);

    auto* overlay = d.defineBooleanParam("crv_enable");
    overlay->setLabel("Tonescale Overlay");
    overlay->setDefault(false);
    if (const char* hint = tooltipForParam("crv_enable")) overlay->setHint(hint);
    pOverlay->addChild(*overlay);

    auto* grpAdvancedRoot = d.defineGroupParam("grp_advanced_root"); grpAdvancedRoot->setLabel("Advanced Look Control"); grpAdvancedRoot->setOpen(false);
    auto* grpDisplayMapping = d.defineGroupParam("grp_display_mapping"); grpDisplayMapping->setLabel("Display Mapping"); grpDisplayMapping->setOpen(false); grpDisplayMapping->setParent(*grpAdvancedRoot);
    auto* grpTone = d.defineGroupParam("grp_tonescale"); grpTone->setLabel("Tonescale"); grpTone->setOpen(false); grpTone->setParent(*grpAdvancedRoot);
    auto* grpRender = d.defineGroupParam("grp_render"); grpRender->setLabel("Render Space"); grpRender->setOpen(false); grpRender->setParent(*grpAdvancedRoot);
    auto* grpMidPurity = d.defineGroupParam("grp_mid_purity"); grpMidPurity->setLabel("Mid Purity"); grpMidPurity->setOpen(false); grpMidPurity->setParent(*grpAdvancedRoot);
    auto* grpPurityCompression = d.defineGroupParam("grp_purity_compression"); grpPurityCompression->setLabel("Purity Compression"); grpPurityCompression->setOpen(false); grpPurityCompression->setParent(*grpAdvancedRoot);
    auto* grpBrl = d.defineGroupParam("grp_brl"); grpBrl->setLabel("Brilliance"); grpBrl->setOpen(false); grpBrl->setParent(*grpAdvancedRoot);
    auto* grpHue = d.defineGroupParam("grp_hue"); grpHue->setLabel("Hue"); grpHue->setOpen(false); grpHue->setParent(*grpAdvancedRoot);
    auto* grpDisplay = d.defineGroupParam("grp_display"); grpDisplay->setLabel("Display Overrides"); grpDisplay->setOpen(false); grpDisplay->setParent(*grpAdvancedRoot);

    auto addAdvBool = [&d](const char* n, const char* l, bool def, OFX::GroupParamDescriptor* g){ auto* p=d.defineBooleanParam(n); p->setLabel(l); p->setDefault(def); p->setParent(*g); if (const char* hint = tooltipForParam(n)) p->setHint(hint); return p; };
    auto addAdvD = [&d](const char* n, const char* l, double df, double mn, double mx, OFX::GroupParamDescriptor* g){ auto* p=d.defineDoubleParam(n); p->setLabel(l); p->setDefault(df); p->setRange(mn,mx); p->setDisplayRange(mn,mx); p->setParent(*g); if (const char* hint = tooltipForParam(n)) p->setHint(hint); return p; };
    auto addAdvC = [&d](const char* n, const char* l, int df, const std::vector<const char*>& o, OFX::GroupParamDescriptor* g){ auto* p=d.defineChoiceParam(n); p->setLabel(l); for(auto* s:o)p->appendOption(s); p->setDefault(df); p->setParent(*g); if (const char* hint = tooltipForParam(n)) p->setHint(hint); return p; };

    pAdvanced->addChild(*grpAdvancedRoot);
    addAdvD("tn_Lp", "Peak Luminance", 100.0, 100.0, 1000.0, grpDisplayMapping);
    addAdvD("tn_Lg", "Grey Luminance", 10.0, 3.0, 25.0, grpDisplayMapping);
    addAdvD("tn_gb", "HDR Grey Boost", 0.13, 0.0, 1.0, grpDisplayMapping);
    addAdvD("pt_hdr", "HDR Purity", 0.5, 0.0, 1.0, grpDisplayMapping);

    auto* resetTonescale = d.definePushButtonParam("reset_tonescale");
    resetTonescale->setLabel("Reset Tonescale");
    resetTonescale->setParent(*grpTone);
    addAdvD("tn_con","Contrast",1.66,1.0,2.0,grpTone);
    addAdvD("tn_sh","Shoulder Clip",0.5,0.0,1.0,grpTone);
    addAdvD("tn_toe","Toe",0.003,0.0,0.1,grpTone);
    addAdvD("tn_off","Offset",0.005,0.0,0.02,grpTone);
    addAdvBool("tn_hcon_enable","Enable Contrast High",false,grpTone);
    addAdvD("tn_hcon","Contrast High",0.0,-1.0,1.0,grpTone);
    addAdvD("tn_hcon_pv","Contrast High Pivot",1.0,0.0,4.0,grpTone);
    addAdvD("tn_hcon_st","Contrast High Strength",4.0,0.0,4.0,grpTone);
    addAdvBool("tn_lcon_enable","Enable Contrast Low",false,grpTone);
    addAdvD("tn_lcon","Contrast Low",0.0,0.0,3.0,grpTone);
    addAdvD("tn_lcon_w","Contrast Low Width",0.5,0.0,2.0,grpTone);

    auto* resetRenderSpace = d.definePushButtonParam("reset_render_space");
    resetRenderSpace->setLabel("Reset Render Space");
    resetRenderSpace->setParent(*grpRender);
    addAdvD("rs_sa","Render Space Strength",0.35,0.0,0.6,grpRender);
    addAdvD("rs_rw","Render Space Weight R",0.25,0.0,0.8,grpRender);
    addAdvD("rs_bw","Render Space Weight B",0.55,0.0,0.8,grpRender);

    auto* resetMidPurity = d.definePushButtonParam("reset_mid_purity");
    resetMidPurity->setLabel("Reset Mid Purity");
    resetMidPurity->setParent(*grpMidPurity);
    addAdvBool("ptm_enable","Enable Mid Purity",true,grpMidPurity);
    addAdvD("ptm_low","Mid Purity Low",0.4,0.0,2.0,grpMidPurity);
    addAdvD("ptm_low_rng","Mid Purity Low Range",0.25,0.0,1.0,grpMidPurity);
    addAdvD("ptm_low_st","Mid Purity Low Strength",0.5,0.1,1.0,grpMidPurity);
    addAdvD("ptm_high","Mid Purity High",-0.8,-0.9,0.0,grpMidPurity);
    addAdvD("ptm_high_rng","Mid Purity High Range",0.35,0.0,1.0,grpMidPurity);
    addAdvD("ptm_high_st","Mid Purity High Strength",0.4,0.1,1.0,grpMidPurity);

    auto* resetPurityCompression = d.definePushButtonParam("reset_purity_compression");
    resetPurityCompression->setLabel("Reset Purity Compression");
    resetPurityCompression->setParent(*grpPurityCompression);
    auto* ptEnable = addAdvBool("pt_enable","Purity Compress High (Always On)",true,grpPurityCompression);
    ptEnable->setEnabled(false);
    addAdvD("pt_lml","Purity Limit Low",0.25,0.0,1.0,grpPurityCompression);
    addAdvD("pt_lml_r","Purity Limit Low R",0.5,0.0,1.0,grpPurityCompression);
    addAdvD("pt_lml_g","Purity Limit Low G",0.0,0.0,1.0,grpPurityCompression);
    addAdvD("pt_lml_b","Purity Limit Low B",0.1,0.0,1.0,grpPurityCompression);
    addAdvD("pt_lmh","Purity Limit High",0.25,0.0,1.0,grpPurityCompression);
    addAdvD("pt_lmh_r","Purity Limit High R",0.5,0.0,1.0,grpPurityCompression);
    addAdvD("pt_lmh_b","Purity Limit High B",0.0,0.0,1.0,grpPurityCompression);
    addAdvBool("ptl_enable","Enable Purity Softclip",true,grpPurityCompression);
    addAdvD("ptl_c","Purity Softclip C",0.06,0.0,0.25,grpPurityCompression);
    addAdvD("ptl_m","Purity Softclip M",0.08,0.0,0.25,grpPurityCompression);
    addAdvD("ptl_y","Purity Softclip Y",0.06,0.0,0.25,grpPurityCompression);

    auto* resetBrilliance = d.definePushButtonParam("reset_brilliance");
    resetBrilliance->setLabel("Reset Brilliance");
    resetBrilliance->setParent(*grpBrl);
    addAdvBool("brl_enable","Enable Brilliance",true,grpBrl);
    addAdvD("brl","Brilliance",0.0,-6.0,2.0,grpBrl);
    addAdvD("brl_r","Brilliance R",-2.5,-6.0,2.0,grpBrl);
    addAdvD("brl_g","Brilliance G",-1.5,-6.0,2.0,grpBrl);
    addAdvD("brl_b","Brilliance B",-1.5,-6.0,2.0,grpBrl);
    addAdvD("brl_rng","Brilliance Range",0.5,0.0,1.0,grpBrl);
    addAdvD("brl_st","Brilliance Strength",0.35,0.0,1.0,grpBrl);
    addAdvBool("brlp_enable","Enable Post Brilliance",true,grpBrl);
    addAdvD("brlp","Brilliance Post",-0.5,-1.0,0.0,grpBrl);
    addAdvD("brlp_r","Post Brilliance R",-1.25,-3.0,0.0,grpBrl);
    addAdvD("brlp_g","Post Brilliance G",-1.25,-3.0,0.0,grpBrl);
    addAdvD("brlp_b","Post Brilliance B",-0.25,-3.0,0.0,grpBrl);

    auto* resetHue = d.definePushButtonParam("reset_hue");
    resetHue->setLabel("Reset Hue");
    resetHue->setParent(*grpHue);
    addAdvBool("hc_enable","Enable Hue Contrast",true,grpHue);
    addAdvD("hc_r","Hue Contrast R",1.0,0.0,2.0,grpHue);
    addAdvD("hc_r_rng","Hue Contrast R Range",0.3,0.0,1.0,grpHue);
    addAdvBool("hs_rgb_enable","Enable Hueshift RGB",true,grpHue);
    addAdvD("hs_r","Hueshift R",0.6,0.0,1.0,grpHue);
    addAdvD("hs_g","Hueshift G",0.35,0.0,1.0,grpHue);
    addAdvD("hs_b","Hueshift B",0.66,0.0,1.0,grpHue);
    addAdvD("hs_r_rng","Hueshift R Range",0.6,0.0,2.0,grpHue);
    addAdvD("hs_g_rng","Hueshift G Range",1.0,0.0,2.0,grpHue);
    addAdvD("hs_b_rng","Hueshift B Range",1.0,0.0,4.0,grpHue);
    addAdvBool("hs_cmy_enable","Enable Hueshift CMY",true,grpHue);
    addAdvD("hs_c","Hueshift C",0.25,0.0,1.0,grpHue);
    addAdvD("hs_m","Hueshift M",0.0,0.0,1.0,grpHue);
    addAdvD("hs_y","Hueshift Y",0.0,0.0,1.0,grpHue);
    addAdvD("hs_c_rng","Hueshift C Range",1.0,0.0,1.0,grpHue);
    addAdvD("hs_m_rng","Hueshift M Range",1.0,0.0,1.0,grpHue);
    addAdvD("hs_y_rng","Hueshift Y Range",1.0,0.0,1.0,grpHue);

    addAdvBool("clamp","Clamp",true,grpDisplay);
    addAdvC("tn_su","Surround",1,{"Dark","Dim","Bright"},grpDisplay);
    addAdvC("display_gamut","Display Gamut",0,{"Rec.709","P3-D65","Rec.2020","P3-D60","P3-DCI","XYZ"},grpDisplay);
    addAdvC("eotf","Display EOTF",2,{"Linear","2.2 Power sRGB","2.4 Power Rec.1886","2.6 Power DCI","ST 2084 PQ","HLG"},grpDisplay);

    auto* userPresetName = d.defineStringParam("userPresetName");
    userPresetName->setLabel("User Preset Name");
    userPresetName->setDefault("");
    userPresetName->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetName);

    auto* userLookSave = d.definePushButtonParam("userLookSave");
    userLookSave->setLabel("Save Look Preset");
    userLookSave->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userLookSave);

    auto* userTonescaleSave = d.definePushButtonParam("userTonescaleSave");
    userTonescaleSave->setLabel("Save Tonescale Preset");
    userTonescaleSave->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userTonescaleSave);

    auto* userPresetImport = d.definePushButtonParam("userPresetImport");
    userPresetImport->setLabel("Import Preset...");
    userPresetImport->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetImport);

    auto* userPresetExportLook = d.definePushButtonParam("userPresetExportLook");
    userPresetExportLook->setLabel("Export Selected Look...");
    userPresetExportLook->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetExportLook);

    auto* userPresetExportTonescale = d.definePushButtonParam("userPresetExportTonescale");
    userPresetExportTonescale->setLabel("Export Selected Tonescale...");
    userPresetExportTonescale->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetExportTonescale);

    auto* userPresetUpdateCurrent = d.definePushButtonParam("userPresetUpdateCurrent");
    userPresetUpdateCurrent->setLabel("Update Current Preset");
    userPresetUpdateCurrent->setEnabled(false);
    userPresetUpdateCurrent->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetUpdateCurrent);

    auto* userPresetDeleteCurrent = d.definePushButtonParam("userPresetDeleteCurrent");
    userPresetDeleteCurrent->setLabel("Delete Current Preset");
    userPresetDeleteCurrent->setEnabled(false);
    userPresetDeleteCurrent->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetDeleteCurrent);

    auto* userPresetRenameCurrent = d.definePushButtonParam("userPresetRenameCurrent");
    userPresetRenameCurrent->setLabel("Rename Current Preset");
    userPresetRenameCurrent->setEnabled(false);
    userPresetRenameCurrent->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetRenameCurrent);

    auto* userPresetRefresh = d.definePushButtonParam("userPresetRefresh");
    userPresetRefresh->setLabel("Refresh Presets");
    userPresetRefresh->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetRefresh);

    auto* pSupport = d.definePageParam("Support");
    pSupport->setLabel("Support");
    auto* grpSupportRoot = d.defineGroupParam("grp_support_root");
    grpSupportRoot->setLabel("Support");
    grpSupportRoot->setOpen(false);
    pSupport->addChild(*grpSupportRoot);

    auto* supportParametersGuide = d.definePushButtonParam("supportParametersGuide");
    supportParametersGuide->setLabel("Parameters Guide");
    supportParametersGuide->setParent(*grpSupportRoot);
    pSupport->addChild(*supportParametersGuide);

    auto* supportLatestReleases = d.definePushButtonParam("supportLatestReleases");
    supportLatestReleases->setLabel("Latest Releases");
    supportLatestReleases->setParent(*grpSupportRoot);
    pSupport->addChild(*supportLatestReleases);

    auto* supportReportIssue = d.definePushButtonParam("supportReportIssue");
    supportReportIssue->setLabel("Report an Issue");
    supportReportIssue->setParent(*grpSupportRoot);
    pSupport->addChild(*supportReportIssue);

    // Keep version labels at the bottom of the Support tab for quick reference.
    auto* supportPortedVersion = d.defineStringParam("supportPortedVersion");
    supportPortedVersion->setLabel("Ported from version");
    supportPortedVersion->setDefault("V1.1.0");
    supportPortedVersion->setEnabled(false);
    supportPortedVersion->setParent(*grpSupportRoot);
    pSupport->addChild(*supportPortedVersion);

    auto* supportOfxVersion = d.defineStringParam("supportOfxVersion");
    supportOfxVersion->setLabel("OFX version");
    supportOfxVersion->setDefault("v1.1.0-rc1");
    supportOfxVersion->setEnabled(false);
    supportOfxVersion->setParent(*grpSupportRoot);
    pSupport->addChild(*supportOfxVersion);
  }

  OFX::ImageEffect* createInstance(OfxImageEffectHandle h, OFX::ContextEnum) override {
    return new OpenDRTEffect(h);
  }
};

}  // namespace

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& ids) {
  static OpenDRTFactory p;
  ids.push_back(&p);
}


















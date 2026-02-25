#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <mutex>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ofxsImageEffect.h"

#include "OpenDRTParams.h"
#include "OpenDRTPresets.h"
#include "OpenDRTProcessor.h"

#define kPluginName "ME_OpenDRT"
#define kPluginGrouping "Color"
#define kPluginDescription "OpenDRT v1.1.0 by Jed Smith, ported to OFX by Moaz ELgabry"
#define kPluginIdentifier "com.moazelgabry.me_opendrt"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 3

namespace {

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

void perfLog(const char* stage, const std::chrono::steady_clock::time_point& start) {
  if (!perfLogEnabled()) return;
  const auto now = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(now - start).count();
  std::fprintf(stderr, "[ME_OpenDRT][PERF] %s: %.3f ms\n", stage, ms);
}

constexpr int kBuiltInLookPresetCount = static_cast<int>(kLookPresetNames.size());
// User preset slot count (look + tonescale). Update this value to add/remove slots.
// If changed, only UI loops below should need adjustment automatically.
constexpr int kUserLookPresetSlotCount = 4;
constexpr int kBuiltInTonescalePresetCount = static_cast<int>(kTonescalePresetNames.size());
constexpr int kUserTonescalePresetSlotCount = 4;

struct UserLookSlot {
  bool used = false;
  std::string name;
  LookPresetValues values{};
};

struct UserTonescaleSlot {
  bool used = false;
  std::string name;
  TonescalePresetValues values{};
};

struct UserPresetStore {
  bool loaded = false;
  std::array<UserLookSlot, kUserLookPresetSlotCount> lookSlots{};
  std::array<UserTonescaleSlot, kUserTonescalePresetSlotCount> tonescaleSlots{};
  std::vector<int> visibleLookSlots;
  std::vector<int> visibleTonescaleSlots;
};

UserPresetStore& userPresetStore() {
  static UserPresetStore store;
  return store;
}

std::mutex& userPresetMutex() {
  static std::mutex m;
  return m;
}

std::filesystem::path userPresetFilePath() {
#ifdef _WIN32
  const char* base = std::getenv("APPDATA");
  if (!base || !*base) base = std::getenv("LOCALAPPDATA");
  if (base && *base) {
    return std::filesystem::path(base) / "ME_OpenDRT" / "user_presets_v1.txt";
  }
#else
  const char* home = std::getenv("HOME");
  if (home && *home) {
    return std::filesystem::path(home) / "Library" / "Application Support" / "ME_OpenDRT" / "user_presets_v1.txt";
  }
#endif
  return std::filesystem::path("ME_OpenDRT_user_presets_v1.txt");
}

std::string sanitizePresetName(const std::string& s, const char* fallback) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\n' || c == '\r' || c == '\t' || c == '|') continue;
    out.push_back(c);
  }
  while (!out.empty() && out.front() == ' ') out.erase(out.begin());
  while (!out.empty() && out.back() == ' ') out.pop_back();
  if (out.empty()) out = fallback;
  if (out.size() > 64) out.resize(64);
  return out;
}

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

void saveUserPresetStoreLocked() {
  const auto path = userPresetFilePath();
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if (!os.is_open()) return;
  os << "ME_OPENDRT_USER_PRESETS_V1\n";

  UserPresetStore& s = userPresetStore();
  for (int i : s.visibleLookSlots) {
    if (i < 0 || i >= kUserLookPresetSlotCount) continue;
    if (!s.lookSlots[static_cast<size_t>(i)].used) continue;
    std::string values;
    serializeLookValues(s.lookSlots[static_cast<size_t>(i)].values, values);
    os << "LOOK\t" << i << '\t' << sanitizePresetName(s.lookSlots[static_cast<size_t>(i)].name, "User Look") << '\t' << values << '\n';
  }
  for (int i : s.visibleTonescaleSlots) {
    if (i < 0 || i >= kUserTonescalePresetSlotCount) continue;
    if (!s.tonescaleSlots[static_cast<size_t>(i)].used) continue;
    std::string values;
    serializeTonescaleValues(s.tonescaleSlots[static_cast<size_t>(i)].values, values);
    os << "TONE\t" << i << '\t' << sanitizePresetName(s.tonescaleSlots[static_cast<size_t>(i)].name, "User Tonescale") << '\t' << values << '\n';
  }
}

void ensureUserPresetStoreLoadedLocked() {
  UserPresetStore& s = userPresetStore();
  if (s.loaded) return;
  s = UserPresetStore{};
  s.loaded = true;

  std::ifstream is(userPresetFilePath(), std::ios::binary);
  if (!is.is_open()) return;

  std::string header;
  std::getline(is, header);
  if (header != "ME_OPENDRT_USER_PRESETS_V1") return;

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
    const int slot = std::atoi(line.substr(p1 + 1, p2 - p1 - 1).c_str());
    const std::string name = sanitizePresetName(line.substr(p2 + 1, p3 - p2 - 1), "User Preset");
    const std::string values = line.substr(p3 + 1);

    if (kind == "LOOK" && slot >= 0 && slot < kUserLookPresetSlotCount) {
      LookPresetValues parsed{};
      if (parseLookValues(values, &parsed)) {
        auto& dst = s.lookSlots[static_cast<size_t>(slot)];
        dst.used = true;
        dst.name = name;
        dst.values = parsed;
        bool exists = false;
        for (int v : s.visibleLookSlots) if (v == slot) { exists = true; break; }
        if (!exists) s.visibleLookSlots.push_back(slot);
      }
    } else if (kind == "TONE" && slot >= 0 && slot < kUserTonescalePresetSlotCount) {
      TonescalePresetValues parsed{};
      if (parseTonescaleValues(values, &parsed)) {
        auto& dst = s.tonescaleSlots[static_cast<size_t>(slot)];
        dst.used = true;
        dst.name = name;
        dst.values = parsed;
        bool exists = false;
        for (int v : s.visibleTonescaleSlots) if (v == slot) { exists = true; break; }
        if (!exists) s.visibleTonescaleSlots.push_back(slot);
      }
    }
  }
}

void rebuildVisiblePresetMapsLocked() {
  UserPresetStore& s = userPresetStore();
  std::vector<int> look;
  for (int v : s.visibleLookSlots) {
    if (v >= 0 && v < kUserLookPresetSlotCount && s.lookSlots[static_cast<size_t>(v)].used) look.push_back(v);
  }
  for (int i = 0; i < kUserLookPresetSlotCount; ++i) {
    if (!s.lookSlots[static_cast<size_t>(i)].used) continue;
    bool exists = false;
    for (int v : look) if (v == i) { exists = true; break; }
    if (!exists) look.push_back(i);
  }
  s.visibleLookSlots = look;

  std::vector<int> tone;
  for (int v : s.visibleTonescaleSlots) {
    if (v >= 0 && v < kUserTonescalePresetSlotCount && s.tonescaleSlots[static_cast<size_t>(v)].used) tone.push_back(v);
  }
  for (int i = 0; i < kUserTonescalePresetSlotCount; ++i) {
    if (!s.tonescaleSlots[static_cast<size_t>(i)].used) continue;
    bool exists = false;
    for (int v : tone) if (v == i) { exists = true; break; }
    if (!exists) tone.push_back(i);
  }
  s.visibleTonescaleSlots = tone;
}

bool userLookSlotFromPresetIndex(int idx, int* slotOut) {
  if (!slotOut) return false;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  const int rel = idx - kBuiltInLookPresetCount;
  const auto& map = userPresetStore().visibleLookSlots;
  if (rel < 0 || rel >= static_cast<int>(map.size())) return false;
  *slotOut = map[static_cast<size_t>(rel)];
  return true;
}

int presetIndexFromUserLookSlot(int slot) {
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  const auto& map = userPresetStore().visibleLookSlots;
  for (int i = 0; i < static_cast<int>(map.size()); ++i) {
    if (map[static_cast<size_t>(i)] == slot) return kBuiltInLookPresetCount + i;
  }
  return -1;
}

bool isUserLookPresetIndex(int idx) {
  int slot = -1;
  return userLookSlotFromPresetIndex(idx, &slot);
}

bool userTonescaleSlotFromPresetIndex(int idx, int* slotOut) {
  if (!slotOut) return false;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  const int rel = idx - kBuiltInTonescalePresetCount;
  const auto& map = userPresetStore().visibleTonescaleSlots;
  if (rel < 0 || rel >= static_cast<int>(map.size())) return false;
  *slotOut = map[static_cast<size_t>(rel)];
  return true;
}

int presetIndexFromUserTonescaleSlot(int slot) {
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  const auto& map = userPresetStore().visibleTonescaleSlots;
  for (int i = 0; i < static_cast<int>(map.size()); ++i) {
    if (map[static_cast<size_t>(i)] == slot) return kBuiltInTonescalePresetCount + i;
  }
  return -1;
}

bool isUserTonescalePresetIndex(int idx) {
  int slot = -1;
  return userTonescaleSlotFromPresetIndex(idx, &slot);
}

std::vector<std::string> visibleUserLookNames() {
  std::vector<std::string> out;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  for (int slot : userPresetStore().visibleLookSlots) {
    const auto& s = userPresetStore().lookSlots[static_cast<size_t>(slot)];
    out.push_back(s.used ? s.name : ("User Look Slot " + std::to_string(slot + 1)));
  }
  return out;
}

std::vector<std::string> visibleUserTonescaleNames() {
  std::vector<std::string> out;
  std::lock_guard<std::mutex> lock(userPresetMutex());
  ensureUserPresetStoreLoadedLocked();
  for (int slot : userPresetStore().visibleTonescaleSlots) {
    const auto& s = userPresetStore().tonescaleSlots[static_cast<size_t>(slot)];
    out.push_back(s.used ? s.name : ("User Tonescale Slot " + std::to_string(slot + 1)));
  }
  return out;
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

const char* tooltipForParam(const std::string& name) {
  static const std::unordered_map<std::string, const char*> kTooltips = {
    {"tn_Lp", "Peak display luminance in nits."},
    {"tn_gb", "Amount of stops to boost grey luminance per stop of peak luminance increase."},
    {"pt_hdr", "How much purity compression and hue shift behavior changes as peak luminance increases."},
    {"tn_Lg", "Display luminance target for middle grey (0.18) in nits."},
    {"lookPreset", "Choose a preset look."},
    {"tonescalePreset", "Choose a tonescale preset or keep the look preset tonescale."},
    {"creativeWhitePreset", "Set the creative whitepoint of display peak luminance."},
    {"cwp_lm", "Limit the intensity range affected by Creative Whitepoint."},
    {"displayEncodingPreset", "Choose the target viewing environment."},
    {"tn_con", "Adjust contrast/slope in display linear."},
    {"tn_sh", "Controls where tonescale crosses display peak and clips."},
    {"tn_toe", "Quadratic toe compression for deep shadows."},
    {"tn_off", "Pre-tonescale scene-linear offset."},
    {"tn_hcon_enable", "Enable upper-tonescale contrast adjustment."},
    {"tn_hcon", "Highlight contrast amount."},
    {"tn_hcon_pv", "Stops above middle grey where highlight adjustment starts."},
    {"tn_hcon_st", "How quickly highlight adjustment ramps in."},
    {"tn_lcon_enable", "Enable low/mid contrast adjustment."},
    {"tn_lcon", "Low-contrast amount."},
    {"tn_lcon_w", "Low-contrast width."},
    {"rs_sa", "Render-space desaturation strength."},
    {"rs_rw", "Red weight for render-space scaling."},
    {"rs_bw", "Blue weight for render-space scaling."},
    {"pt_enable", "Compresses purity as intensity increases."},
    {"pt_lml", "Purity compression limit as intensity decreases (all hues)."},
    {"pt_lml_r", "Purity compression limit as intensity decreases (reds)."},
    {"pt_lml_g", "Purity compression limit as intensity decreases (greens)."},
    {"pt_lml_b", "Purity compression limit as intensity decreases (blues)."},
    {"pt_lmh", "Purity compression limit as intensity increases (all hues)."},
    {"pt_lmh_r", "Purity compression limit as intensity increases (reds)."},
    {"pt_lmh_b", "Purity compression limit as intensity increases (blues)."},
    {"ptl_enable", "Enable purity softclip."},
    {"ptl_c", "Purity softclip strength for cyan."},
    {"ptl_m", "Purity softclip strength for magenta."},
    {"ptl_y", "Purity softclip strength for yellow."},
    {"ptm_enable", "Enable mid-range purity adjustments."},
    {"ptm_low", "Increase mid-range purity in low/mid intensities."},
    {"ptm_low_rng", "Range for mid-purity low adjustment."},
    {"ptm_low_st", "Strength curve for mid-purity low adjustment."},
    {"ptm_high", "Decrease mid-range purity in upper-mid/high intensities."},
    {"ptm_high_rng", "Range for mid-purity high adjustment."},
    {"ptm_high_st", "Strength curve for mid-purity high adjustment."},
    {"brl_enable", "Enable brilliance module."},
    {"brl", "Global brilliance amount."},
    {"brl_r", "Brilliance adjustment for red."},
    {"brl_g", "Brilliance adjustment for green."},
    {"brl_b", "Brilliance adjustment for blue."},
    {"brl_rng", "Brilliance intensity-range weighting."},
    {"brl_st", "Brilliance purity-range weighting."},
    {"brlp_enable", "Enable post-brilliance module."},
    {"brlp", "Global post-brilliance amount."},
    {"brlp_r", "Post-brilliance adjustment for red."},
    {"brlp_g", "Post-brilliance adjustment for green."},
    {"brlp_b", "Post-brilliance adjustment for blue."},
    {"hc_enable", "Enable hue contrast module."},
    {"hc_r", "Hue contrast amount at red hue angle."},
    {"hc_r_rng", "Hue contrast range over intensity."},
    {"hs_rgb_enable", "Enable RGB hue-shift module."},
    {"hs_r", "Red hue-shift amount."},
    {"hs_r_rng", "Red hue-shift range."},
    {"hs_g", "Green hue-shift amount."},
    {"hs_g_rng", "Green hue-shift range."},
    {"hs_b", "Blue hue-shift amount."},
    {"hs_b_rng", "Blue hue-shift range."},
    {"hs_cmy_enable", "Enable CMY hue-shift module."},
    {"hs_c", "Cyan hue-shift amount."},
    {"hs_c_rng", "Cyan hue-shift range."},
    {"hs_m", "Magenta hue-shift amount."},
    {"hs_m_rng", "Magenta hue-shift range."},
    {"hs_y", "Yellow hue-shift amount."},
    {"hs_y_rng", "Yellow hue-shift range."},
    {"clamp", "Clamp final image to display-supported range."},
    {"tn_su", "Surround compensation mode."},
    {"display_gamut", "Target display gamut."},
    {"eotf", "Target display transfer function."},
    {"crv_enable", "Draw tonescale overlay."}
  };
  auto it = kTooltips.find(name);
  return it == kTooltips.end() ? nullptr : it->second;
}

class OpenDRTEffect : public OFX::ImageEffect {
 public:
  explicit OpenDRTEffect(OfxImageEffectHandle handle)
      : ImageEffect(handle) {
    dstClip_ = fetchClip(kOfxImageEffectOutputClipName);
    srcClip_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    refreshUserSlotLabels();
    updateToggleVisibility(0.0);
  }

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

    bool rendered = false;
    if (!forceStageCopyEnabled() && srcLayout.valid && dstLayout.valid) {
      const auto tBackendDirect = std::chrono::steady_clock::now();
      rendered = processor_->renderWithLayout(
          srcLayout.base, dstLayout.base, width, height, srcLayout.pitchBytes, dstLayout.pitchBytes, true, false);
      perfLog("Backend render direct", tBackendDirect);
    }

    if (!rendered) {
      const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
      if (srcPixels_.size() != pixelCount) {
        srcPixels_.assign(pixelCount, 0.0f);
        dstPixels_.assign(pixelCount, 0.0f);
      }

      const auto tStageCopyStart = std::chrono::steady_clock::now();
      if (srcLayout.valid && srcLayout.contiguous) {
        std::memcpy(srcPixels_.data(), srcLayout.base, rowBytes * static_cast<size_t>(height));
      } else {
        // Row fallback for hosts with non-contiguous row layout.
        for (int y = bounds.y1; y < bounds.y2; ++y) {
          const int localY = y - bounds.y1;
          float* sp = static_cast<float*>(src->getPixelAddress(bounds.x1, y));
          float* rowDst = srcPixels_.data() + static_cast<size_t>(localY) * static_cast<size_t>(width) * 4u;
          if (sp != nullptr) {
            std::memcpy(rowDst, sp, rowBytes);
          } else {
            std::memset(rowDst, 0, rowBytes);
          }
        }
      }
      perfLog("Host src staging", tStageCopyStart);

      const auto tBackendStart = std::chrono::steady_clock::now();
      rendered = processor_->render(srcPixels_.data(), dstPixels_.data(), width, height, true, false);
      perfLog("Backend render staging", tBackendStart);
      if (!rendered) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
      }

      const auto tDstCopyStart = std::chrono::steady_clock::now();
      if (dstLayout.valid && dstLayout.contiguous) {
        std::memcpy(dstLayout.base, dstPixels_.data(), rowBytes * static_cast<size_t>(height));
      } else {
        for (int y = bounds.y1; y < bounds.y2; ++y) {
          const int localY = y - bounds.y1;
          float* dp = static_cast<float*>(dst->getPixelAddress(bounds.x1, y));
          if (!dp) continue;
          const float* rowSrc = dstPixels_.data() + static_cast<size_t>(localY) * static_cast<size_t>(width) * 4u;
          std::memcpy(dp, rowSrc, rowBytes);
        }
      }
      perfLog("Host dst copy", tDstCopyStart);
    }

    perfLog("Render total", tRenderStart);
  }

  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override {
    try {
      if (suppressParamChanged_) {
        return;
      }
      if (args.reason == OFX::eChangePluginEdit || args.reason == OFX::eChangeTime) {
        return;
      }

      if (paramName == "presetState" || paramName == "baseWhitepointLabel") {
        return;
      }
      const bool isUserLabel =
        ((paramName.rfind("userLookSlot", 0) == 0 || paramName.rfind("userToneSlot", 0) == 0) &&
         paramName.find("Label") != std::string::npos);
      if (paramName == "surroundLabel" ||
          paramName == "activeUserLookSlot" || paramName == "activeUserToneSlot" || isUserLabel) {
        return;
      }

      if (paramName == "lookPreset") {
        int look = getChoice("lookPreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        setChoice("tonescalePreset", 0);
        setChoice("creativeWhitePreset", 0);
        setInt("activeUserLookSlot", -1);
        setInt("activeUserToneSlot", -1);
        if (isUserLookPresetIndex(look)) {
          int slot = -1;
          if (!userLookSlotFromPresetIndex(look, &slot)) return;
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          const auto& userSlot = userPresetStore().lookSlots[static_cast<size_t>(slot)];
          if (userSlot.used) {
            writeLookValuesToParams(userSlot.values, *this);
            setString("baseWhitepointLabel", whitepointNameFromCwp(userSlot.values.cwp));
          } else {
            setString("baseWhitepointLabel", "D65");
          }
        } else {
          writePresetToParams(look, *this);
        }
        updateToggleVisibility(args.time);
        setInt("presetState", 0);
        setString("baseWhitepointLabel", effectiveWhitepointLabel(look, 0));
        return;
      }

      if (paramName == "tonescalePreset") {
        const int look = getChoice("lookPreset", args.time, 0);
        const int tsPreset = getChoice("tonescalePreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        setInt("activeUserToneSlot", -1);
        if (isUserTonescalePresetIndex(tsPreset)) {
          int slot = -1;
          if (!userTonescaleSlotFromPresetIndex(tsPreset, &slot)) return;
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          const auto& userSlot = userPresetStore().tonescaleSlots[static_cast<size_t>(slot)];
          if (userSlot.used) {
            writeTonescaleValuesToParams(userSlot.values, *this);
          }
        } else {
          writeTonescalePresetToParams(tsPreset, *this);
        }
        updateToggleVisibility(args.time);
        updatePresetStateFromCurrent(args.time);
        return;
      }

      if (paramName == "creativeWhitePreset") {
        const int look = getChoice("lookPreset", args.time, 0);
        const int cwpPreset = getChoice("creativeWhitePreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        writeCreativeWhitePresetToParams(cwpPreset, *this);
        setString("baseWhitepointLabel", effectiveWhitepointLabel(look, cwpPreset));
        updatePresetStateFromCurrent(args.time);
        return;
      }

      if (paramName == "displayEncodingPreset") {
        int preset = getChoice("displayEncodingPreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        writeDisplayPresetToParams(preset, *this);
        updatePresetStateFromCurrent(args.time);
        return;
      }

      if (paramName == "userLookSave") {
        const int slot = getChoice("userLookSlotSelect", args.time, 0);
        const std::string typedName = getString("userPresetName", "User Look");
        const std::string name = sanitizePresetName(typedName, "User Look");
        const LookPresetValues values = captureCurrentLookValues(args.time);
        bool existed = false;
        {
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          auto& dst = userPresetStore().lookSlots[static_cast<size_t>(slot)];
          existed = dst.used;
          dst.used = true;
          dst.name = name;
          dst.values = values;
          rebuildVisiblePresetMapsLocked();
          saveUserPresetStoreLocked();
        }
        FlagScope scope(suppressParamChanged_);
        setInt("activeUserLookSlot", slot);
        ensureLookSlotMenuVisible(slot, name, existed);
        refreshUserSlotLabels();
        const int idx = presetIndexFromUserLookSlot(slot);
        if (idx >= 0) setChoice("lookPreset", idx);
        writeLookValuesToParams(values, *this);
        updateToggleVisibility(args.time);
        updatePresetStateFromCurrent(args.time);
        return;
      }

      if (paramName == "userLookLoad") {
        const int slot = getChoice("userLookSlotSelect", args.time, 0);
        bool used = false;
        LookPresetValues values{};
        std::string name;
        {
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          const auto& src = userPresetStore().lookSlots[static_cast<size_t>(slot)];
          used = src.used;
          if (used) {
            values = src.values;
            name = src.name;
          }
        }
        if (used) {
          FlagScope scope(suppressParamChanged_);
          setInt("activeUserLookSlot", slot);
          ensureLookSlotMenuVisible(slot, name, true);
          const int idx = presetIndexFromUserLookSlot(slot);
          if (idx >= 0) setChoice("lookPreset", idx);
          writeLookValuesToParams(values, *this);
          updateToggleVisibility(args.time);
          updatePresetStateFromCurrent(args.time);
        }
        return;
      }

      if (paramName == "userTonescaleSave") {
        const int slot = getChoice("userToneSlotSelect", args.time, 0);
        const std::string typedName = getString("userPresetName", "User Tonescale");
        const std::string name = sanitizePresetName(typedName, "User Tonescale");
        const TonescalePresetValues values = captureCurrentTonescaleValues(args.time);
        bool existed = false;
        {
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          auto& dst = userPresetStore().tonescaleSlots[static_cast<size_t>(slot)];
          existed = dst.used;
          dst.used = true;
          dst.name = name;
          dst.values = values;
          rebuildVisiblePresetMapsLocked();
          saveUserPresetStoreLocked();
        }
        FlagScope scope(suppressParamChanged_);
        setInt("activeUserToneSlot", slot);
        ensureTonescaleSlotMenuVisible(slot, name, existed);
        refreshUserSlotLabels();
        const int idx = presetIndexFromUserTonescaleSlot(slot);
        if (idx >= 0) setChoice("tonescalePreset", idx);
        writeTonescaleValuesToParams(values, *this);
        updateToggleVisibility(args.time);
        updatePresetStateFromCurrent(args.time);
        return;
      }

      if (paramName == "userTonescaleLoad") {
        const int slot = getChoice("userToneSlotSelect", args.time, 0);
        bool used = false;
        TonescalePresetValues values{};
        std::string name;
        {
          std::lock_guard<std::mutex> lock(userPresetMutex());
          ensureUserPresetStoreLoadedLocked();
          const auto& src = userPresetStore().tonescaleSlots[static_cast<size_t>(slot)];
          used = src.used;
          if (used) {
            values = src.values;
            name = src.name;
          }
        }
        if (used) {
          FlagScope scope(suppressParamChanged_);
          setInt("activeUserToneSlot", slot);
          ensureTonescaleSlotMenuVisible(slot, name, true);
          const int idx = presetIndexFromUserTonescaleSlot(slot);
          if (idx >= 0) setChoice("tonescalePreset", idx);
          writeTonescaleValuesToParams(values, *this);
          updateToggleVisibility(args.time);
          updatePresetStateFromCurrent(args.time);
        }
        return;
      }

      if (isAdvancedParam(paramName)) {
        FlagScope scope(suppressParamChanged_);
        if (isVisibilityToggleParam(paramName)) {
          updateToggleVisibility(args.time);
        }
        if (paramName == "tn_su") setString("surroundLabel", surroundNameFromIndex(getChoice("tn_su", args.time, 1)));
        updatePresetStateFromCurrent(args.time);
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

  bool isAdvancedParam(const std::string& name) const {
    static const std::vector<std::string> names = {
      "tn_con","tn_sh","tn_toe","tn_off","tn_hcon_enable","tn_hcon","tn_hcon_pv","tn_hcon_st","tn_lcon_enable","tn_lcon","tn_lcon_w",
      "rs_sa","rs_rw","rs_bw",
      "pt_enable","pt_lml","pt_lml_r","pt_lml_g","pt_lml_b","pt_lmh","pt_lmh_r","pt_lmh_b","ptl_enable","ptl_c","ptl_m","ptl_y",
      "ptm_enable","ptm_low","ptm_low_rng","ptm_low_st","ptm_high","ptm_high_rng","ptm_high_st",
      "brl_enable","brl","brl_r","brl_g","brl_b","brl_rng","brl_st","brlp_enable","brlp","brlp_r","brlp_g","brlp_b",
      "hc_enable","hc_r","hc_r_rng","hs_rgb_enable","hs_r","hs_r_rng","hs_g","hs_g_rng","hs_b","hs_b_rng","hs_cmy_enable","hs_c","hs_c_rng","hs_m","hs_m_rng","hs_y","hs_y_rng",
      "clamp","tn_su","display_gamut","eotf"
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

  std::string lookPresetDisplayName(int lookPresetIndex) const {
    if (!isUserLookPresetIndex(lookPresetIndex)) {
      return currentPresetName(lookPresetIndex);
    }
    int slot = -1;
    if (!userLookSlotFromPresetIndex(lookPresetIndex, &slot)) return std::string("Unknown User Look");
    std::lock_guard<std::mutex> lock(userPresetMutex());
    ensureUserPresetStoreLoadedLocked();
    const auto& s = userPresetStore().lookSlots[static_cast<size_t>(slot)];
    if (s.used && !s.name.empty()) return s.name;
    return std::string("User Look Slot ") + std::to_string(slot + 1);
  }

  std::string presetLabelCleanForLook(int lookPresetIndex) const {
    return lookPresetDisplayName(lookPresetIndex) + " | " + buildLabelText();
  }

  std::string presetLabelCustomForLook(int lookPresetIndex) const {
    return std::string("Custom (") + lookPresetDisplayName(lookPresetIndex) + ") | " + buildLabelText();
  }

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

  void refreshUserSlotLabels() {
    std::lock_guard<std::mutex> lock(userPresetMutex());
    ensureUserPresetStoreLoadedLocked();
    for (int i = 0; i < kUserLookPresetSlotCount; ++i) {
      const auto& s = userPresetStore().lookSlots[static_cast<size_t>(i)];
      const std::string fallback = std::string("Empty");
      setString(("userLookSlot" + std::to_string(i + 1) + "Label").c_str(), s.used ? s.name : fallback);
    }
    for (int i = 0; i < kUserTonescalePresetSlotCount; ++i) {
      const auto& s = userPresetStore().tonescaleSlots[static_cast<size_t>(i)];
      const std::string fallback = std::string("Empty");
      setString(("userToneSlot" + std::to_string(i + 1) + "Label").c_str(), s.used ? s.name : fallback);
    }
  }

  bool isCurrentEqualToPresetBaseline(double time, bool* tonescaleCleanOut = nullptr) const {
    const int look = getChoice("lookPreset", time, 0);
    const int tsPreset = getChoice("tonescalePreset", time, 0);
    const int displayPreset = getChoice("displayEncodingPreset", time, 0);
    const int cwpPreset = getChoice("creativeWhitePreset", time, 0);
    const int activeUserLookSlot = getInt("activeUserLookSlot", time, -1);
    const int activeUserToneSlot = getInt("activeUserToneSlot", time, -1);

    OpenDRTParams expected{};
    if (activeUserLookSlot >= 0 && activeUserLookSlot < kUserLookPresetSlotCount) {
      std::lock_guard<std::mutex> lock(userPresetMutex());
      ensureUserPresetStoreLoadedLocked();
      const auto& s = userPresetStore().lookSlots[static_cast<size_t>(activeUserLookSlot)];
      if (!s.used) return false;
      applyLookValuesToResolved(expected, s.values);
    } else {
      applyLookPresetToResolved(expected, look);
    }

    if (activeUserToneSlot >= 0 && activeUserToneSlot < kUserTonescalePresetSlotCount) {
      std::lock_guard<std::mutex> lock(userPresetMutex());
      ensureUserPresetStoreLoadedLocked();
      const auto& s = userPresetStore().tonescaleSlots[static_cast<size_t>(activeUserToneSlot)];
      if (!s.used) return false;
      applyTonescaleValuesToResolved(expected, s.values);
    } else {
      applyTonescalePresetToResolved(expected, tsPreset);
    }

    applyDisplayEncodingPreset(expected, displayPreset);
    if (cwpPreset > 0) expected.cwp = cwpPreset - 1;

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
      (getBool("clamp", time, expected.clamp) == expected.clamp) &&
      (getChoice("tn_su", time, expected.tn_su) == expected.tn_su) &&
      (getChoice("display_gamut", time, expected.display_gamut) == expected.display_gamut) &&
      (getChoice("eotf", time, expected.eotf) == expected.eotf);

    return clean;
  }

  void updatePresetStateFromCurrent(double time) {
    const int look = getChoice("lookPreset", time, 0);
    const int tsPreset = getChoice("tonescalePreset", time, 0);
    const int cwpPreset = getChoice("creativeWhitePreset", time, 0);
    const int activeUserLookSlot = getInt("activeUserLookSlot", time, -1);
    const int activeUserToneSlot = getInt("activeUserToneSlot", time, -1);
    bool tonescaleClean = true;
    const bool clean = isCurrentEqualToPresetBaseline(time, &tonescaleClean);
    setInt("presetState", clean ? 0 : 1);
    (void)activeUserToneSlot;
    (void)tonescaleClean;
    if (activeUserLookSlot >= 0 && activeUserLookSlot < kUserLookPresetSlotCount) {
      {
        std::lock_guard<std::mutex> lock(userPresetMutex());
        ensureUserPresetStoreLoadedLocked();
        const auto& s = userPresetStore().lookSlots[static_cast<size_t>(activeUserLookSlot)];
        if (s.used) setString("baseWhitepointLabel", whitepointNameFromCwp(s.values.cwp));
      }
    } else {
      setString("baseWhitepointLabel", effectiveWhitepointLabel(look, cwpPreset));
    }
  }

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
  void setChoice(const char* name, int v) {
    if (auto* p = fetchChoiceParam(name)) p->setValue(v);
  }
  void ensureLookSlotMenuVisible(int slot, const std::string& name, bool existed) {
    if (slot < 0 || slot >= kUserLookPresetSlotCount) return;
    const int idx = presetIndexFromUserLookSlot(slot);
    if (auto* p = fetchChoiceParam("lookPreset")) {
      if (idx >= 0 && existed) p->setOption(idx, name);
      else p->appendOption(name);
    }
  }
  void ensureTonescaleSlotMenuVisible(int slot, const std::string& name, bool existed) {
    if (slot < 0 || slot >= kUserTonescalePresetSlotCount) return;
    const int idx = presetIndexFromUserTonescaleSlot(slot);
    if (auto* p = fetchChoiceParam("tonescalePreset")) {
      if (idx >= 0 && existed) p->setOption(idx, name);
      else p->appendOption(name);
    }
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
};

class OpenDRTFactory : public OFX::PluginFactoryHelper<OpenDRTFactory> {
 public:
  OpenDRTFactory() : PluginFactoryHelper<OpenDRTFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}

  void load() override {}
  void unload() override {}

  void describe(OFX::ImageEffectDescriptor& d) override {
    static const std::string nameWithVersion = "ME_OpenDRT 1.1.0 v1.0";
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
    d.setSupportsCudaRender(false);
    d.setSupportsCudaStream(false);
  }

  void describeInContext(OFX::ImageEffectDescriptor& d, OFX::ContextEnum) override {
    OFX::ClipDescriptor* src = d.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);

    OFX::ClipDescriptor* dst = d.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    auto* pInput = d.definePageParam("Input");
    auto* pLook = d.definePageParam("Look");
    auto* pTonescale = d.definePageParam("Tonescale");
    auto* pAdvanced = d.definePageParam("Advanced Look Control");
    auto* pOverlay = d.definePageParam("Overlay");
    auto* pUserPresets = d.definePageParam("User Presets");
    auto* grpUserPresetsRoot = d.defineGroupParam("grp_user_presets_root");
    grpUserPresetsRoot->setLabel("User Presets");
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
    pInput->addChild(*inGamut); pInput->addChild(*inOetf);

    auto* dep = addChoice("displayEncodingPreset", "Display Encoding Preset", 0, {"Rec.1886 - 2.4 Power / Rec.709","sRGB Display - 2.2 Power / Rec.709","Display P3 - 2.2 Power / P3-D65","DCI - 2.6 Power / P3-D60","DCI - 2.6 Power / P3-DCI","DCI - 2.6 Power / XYZ","Rec.2100 - PQ / Rec.2020","Rec.2100 - HLG / Rec.2020","Dolby - PQ / P3-D65"});
    auto* lookPreset = addChoice("lookPreset", "Look Preset", 0, {"Standard","Arriba","Sylvan","Colorful","Aery","Dystopic","Umbra","Base"});
    for (const auto& n : visibleUserLookNames()) lookPreset->appendOption(n);
    auto* baseWpLabel = d.defineStringParam("baseWhitepointLabel"); baseWpLabel->setLabel("Base Whitepoint"); baseWpLabel->setDefault(baseWhitepointLabelForLook(0)); baseWpLabel->setEnabled(false);
    auto* surroundLabel = d.defineStringParam("surroundLabel"); surroundLabel->setLabel("Selected Surround"); surroundLabel->setDefault("Dim"); surroundLabel->setEnabled(false);
    auto* presetState = d.defineIntParam("presetState"); presetState->setIsSecret(true); presetState->setDefault(0);
    auto* cwpHidden = d.defineIntParam("cwp"); cwpHidden->setIsSecret(true); cwpHidden->setDefault(2);
    auto* activeUserLookSlot = d.defineIntParam("activeUserLookSlot"); activeUserLookSlot->setIsSecret(true); activeUserLookSlot->setDefault(-1);
    auto* activeUserToneSlot = d.defineIntParam("activeUserToneSlot"); activeUserToneSlot->setIsSecret(true); activeUserToneSlot->setDefault(-1);
    auto* tonescalePreset = addChoice("tonescalePreset", "Tonescale Preset", 0, {"USE LOOK PRESET","Low Contrast","Medium Contrast","High Contrast","Arriba Tonescale","Sylvan Tonescale","Colorful Tonescale","Aery Tonescale","Dystopic Tonescale","Umbra Tonescale","ACES-1.x","ACES-2.0","Marvelous Tonescape","DaGrinchi ToneGroan"});
    for (const auto& n : visibleUserTonescaleNames()) tonescalePreset->appendOption(n);
    auto* cwpPreset = addChoice("creativeWhitePreset", "Creative White", 0, {"USE LOOK PRESET","D93","D75","D65","D60","D55","D50"});
    auto* cwpLm = addDouble("cwp_lm", "Creative White Limit", 0.25, 0.0, 1.0);
    pLook->addChild(*dep); pLook->addChild(*lookPreset); pLook->addChild(*baseWpLabel); pLook->addChild(*surroundLabel); pLook->addChild(*presetState); pLook->addChild(*cwpHidden); pLook->addChild(*activeUserLookSlot); pLook->addChild(*activeUserToneSlot); pLook->addChild(*tonescalePreset); pLook->addChild(*cwpPreset); pLook->addChild(*cwpLm);

    pTonescale->addChild(*addDouble("tn_Lp", "Display Peak Luminance", 100.0, 100.0, 1000.0));
    pTonescale->addChild(*addDouble("tn_Lg", "Display Grey Luminance", 10.0, 3.0, 25.0));
    pTonescale->addChild(*addDouble("tn_gb", "HDR Grey Boost", 0.13, 0.0, 1.0));
    pTonescale->addChild(*addDouble("pt_hdr", "HDR Purity", 0.5, 0.0, 1.0));

    auto* grpAdvancedRoot = d.defineGroupParam("grp_advanced_root"); grpAdvancedRoot->setLabel("Advanced"); grpAdvancedRoot->setOpen(false);
    auto* grpTone = d.defineGroupParam("grp_tonescale"); grpTone->setLabel("Tonescale"); grpTone->setOpen(false); grpTone->setParent(*grpAdvancedRoot);
    auto* grpRender = d.defineGroupParam("grp_render"); grpRender->setLabel("Render Space"); grpRender->setOpen(false); grpRender->setParent(*grpAdvancedRoot);
    auto* grpPurity = d.defineGroupParam("grp_purity"); grpPurity->setLabel("Purity"); grpPurity->setOpen(false); grpPurity->setParent(*grpAdvancedRoot);
    auto* grpBrl = d.defineGroupParam("grp_brl"); grpBrl->setLabel("Brilliance"); grpBrl->setOpen(false); grpBrl->setParent(*grpAdvancedRoot);
    auto* grpHue = d.defineGroupParam("grp_hue"); grpHue->setLabel("Hue"); grpHue->setOpen(false); grpHue->setParent(*grpAdvancedRoot);
    auto* grpDisplay = d.defineGroupParam("grp_display"); grpDisplay->setLabel("Display Overrides"); grpDisplay->setOpen(false); grpDisplay->setParent(*grpAdvancedRoot);

    auto addAdvBool = [&d](const char* n, const char* l, bool def, OFX::GroupParamDescriptor* g){ auto* p=d.defineBooleanParam(n); p->setLabel(l); p->setDefault(def); p->setParent(*g); if (const char* hint = tooltipForParam(n)) p->setHint(hint); return p; };
    auto addAdvD = [&d](const char* n, const char* l, double df, double mn, double mx, OFX::GroupParamDescriptor* g){ auto* p=d.defineDoubleParam(n); p->setLabel(l); p->setDefault(df); p->setRange(mn,mx); p->setDisplayRange(mn,mx); p->setParent(*g); if (const char* hint = tooltipForParam(n)) p->setHint(hint); return p; };
    auto addAdvC = [&d](const char* n, const char* l, int df, const std::vector<const char*>& o, OFX::GroupParamDescriptor* g){ auto* p=d.defineChoiceParam(n); p->setLabel(l); for(auto* s:o)p->appendOption(s); p->setDefault(df); p->setParent(*g); if (const char* hint = tooltipForParam(n)) p->setHint(hint); return p; };

    pAdvanced->addChild(*grpAdvancedRoot);

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

    addAdvD("rs_sa","Render Space Strength",0.35,0.0,0.6,grpRender);
    addAdvD("rs_rw","Render Space Weight R",0.25,0.0,0.8,grpRender);
    addAdvD("rs_bw","Render Space Weight B",0.55,0.0,0.8,grpRender);

    auto* ptEnable = addAdvBool("pt_enable","Purity Compress High (Always On)",true,grpPurity);
    ptEnable->setEnabled(false);
    addAdvD("pt_lml","Purity Limit Low",0.25,0.0,1.0,grpPurity);
    addAdvD("pt_lml_r","Purity Limit Low R",0.5,0.0,1.0,grpPurity);
    addAdvD("pt_lml_g","Purity Limit Low G",0.0,0.0,1.0,grpPurity);
    addAdvD("pt_lml_b","Purity Limit Low B",0.1,0.0,1.0,grpPurity);
    addAdvD("pt_lmh","Purity Limit High",0.25,0.0,1.0,grpPurity);
    addAdvD("pt_lmh_r","Purity Limit High R",0.5,0.0,1.0,grpPurity);
    addAdvD("pt_lmh_b","Purity Limit High B",0.0,0.0,1.0,grpPurity);
    addAdvBool("ptl_enable","Enable Purity Softclip",true,grpPurity);
    addAdvD("ptl_c","Purity Softclip C",0.06,0.0,0.25,grpPurity);
    addAdvD("ptl_m","Purity Softclip M",0.08,0.0,0.25,grpPurity);
    addAdvD("ptl_y","Purity Softclip Y",0.06,0.0,0.25,grpPurity);
    addAdvBool("ptm_enable","Enable Mid Purity",true,grpPurity);
    addAdvD("ptm_low","Mid Purity Low",0.4,0.0,2.0,grpPurity);
    addAdvD("ptm_low_rng","Mid Purity Low Range",0.25,0.0,1.0,grpPurity);
    addAdvD("ptm_low_st","Mid Purity Low Strength",0.5,0.1,1.0,grpPurity);
    addAdvD("ptm_high","Mid Purity High",-0.8,-0.9,0.0,grpPurity);
    addAdvD("ptm_high_rng","Mid Purity High Range",0.35,0.0,1.0,grpPurity);
    addAdvD("ptm_high_st","Mid Purity High Strength",0.4,0.1,1.0,grpPurity);

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

    auto* overlay = d.defineBooleanParam("crv_enable");
    overlay->setLabel("Tonescale Overlay");
    overlay->setDefault(false);
    if (const char* hint = tooltipForParam("crv_enable")) overlay->setHint(hint);
    pOverlay->addChild(*overlay);

    auto* userPresetName = d.defineStringParam("userPresetName");
    userPresetName->setLabel("User Preset Name");
    userPresetName->setDefault("");
    userPresetName->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*grpUserPresetsRoot);
    pUserPresets->addChild(*userPresetName);

    auto* userLookSlotSelect = d.defineChoiceParam("userLookSlotSelect");
    userLookSlotSelect->setLabel("Look Slot");
    userLookSlotSelect->appendOption("Slot 1");
    userLookSlotSelect->appendOption("Slot 2");
    userLookSlotSelect->appendOption("Slot 3");
    userLookSlotSelect->appendOption("Slot 4");
    userLookSlotSelect->setDefault(0);
    userLookSlotSelect->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userLookSlotSelect);

    auto* userLookSave = d.definePushButtonParam("userLookSave");
    userLookSave->setLabel("Save Look To Slot");
    userLookSave->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userLookSave);

    auto* userLookLoad = d.definePushButtonParam("userLookLoad");
    userLookLoad->setLabel("Load Look From Slot");
    userLookLoad->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userLookLoad);

    // Slot label params are generated from kUserLookPresetSlotCount. Increase that constant to add slots.
    for (int i = 0; i < kUserLookPresetSlotCount; ++i) {
      const std::string id = "userLookSlot" + std::to_string(i + 1) + "Label";
      const std::string label = "Look Slot " + std::to_string(i + 1) + " Name";
      auto* p = d.defineStringParam(id);
      p->setLabel(label);
      p->setDefault("Empty");
      p->setEnabled(false);
      p->setParent(*grpUserPresetsRoot);
      pUserPresets->addChild(*p);
    }

    auto* userToneSlotSelect = d.defineChoiceParam("userToneSlotSelect");
    userToneSlotSelect->setLabel("Tonescale Slot");
    userToneSlotSelect->appendOption("Slot 1");
    userToneSlotSelect->appendOption("Slot 2");
    userToneSlotSelect->appendOption("Slot 3");
    userToneSlotSelect->appendOption("Slot 4");
    userToneSlotSelect->setDefault(0);
    userToneSlotSelect->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userToneSlotSelect);

    auto* userTonescaleSave = d.definePushButtonParam("userTonescaleSave");
    userTonescaleSave->setLabel("Save Tonescale To Slot");
    userTonescaleSave->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userTonescaleSave);

    auto* userTonescaleLoad = d.definePushButtonParam("userTonescaleLoad");
    userTonescaleLoad->setLabel("Load Tonescale From Slot");
    userTonescaleLoad->setParent(*grpUserPresetsRoot);
    pUserPresets->addChild(*userTonescaleLoad);

    // Slot label params are generated from kUserTonescalePresetSlotCount. Increase that constant to add slots.
    for (int i = 0; i < kUserTonescalePresetSlotCount; ++i) {
      const std::string id = "userToneSlot" + std::to_string(i + 1) + "Label";
      const std::string label = "Tonescale Slot " + std::to_string(i + 1) + " Name";
      auto* p = d.defineStringParam(id);
      p->setLabel(label);
      p->setDefault("Empty");
      p->setEnabled(false);
      p->setParent(*grpUserPresetsRoot);
      pUserPresets->addChild(*p);
    }
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



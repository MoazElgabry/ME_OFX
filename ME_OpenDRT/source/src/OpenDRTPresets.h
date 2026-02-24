#pragma once

#include <array>
#include <string>

#include "ofxsImageEffect.h"

#include "OpenDRTParams.h"

struct OpenDRTRawValues {
  int in_gamut;
  int in_oetf;
  float tn_Lp;
  float tn_gb;
  float pt_hdr;
  float tn_Lg;
  int crv_enable;

  int lookPreset;
  int tonescalePreset;
  int creativeWhitePreset;
  int cwp;
  float creativeWhiteLimit;
  int displayEncodingPreset;

  float tn_con;
  float tn_sh;
  float tn_toe;
  float tn_off;
  int tn_hcon_enable;
  float tn_hcon;
  float tn_hcon_pv;
  float tn_hcon_st;
  int tn_lcon_enable;
  float tn_lcon;
  float tn_lcon_w;

  float rs_sa;
  float rs_rw;
  float rs_bw;

  int pt_enable;
  float pt_lml;
  float pt_lml_r;
  float pt_lml_g;
  float pt_lml_b;
  float pt_lmh;
  float pt_lmh_r;
  float pt_lmh_b;
  int ptl_enable;
  float ptl_c;
  float ptl_m;
  float ptl_y;
  int ptm_enable;
  float ptm_low;
  float ptm_low_rng;
  float ptm_low_st;
  float ptm_high;
  float ptm_high_rng;
  float ptm_high_st;

  int brl_enable;
  float brl;
  float brl_r;
  float brl_g;
  float brl_b;
  float brl_rng;
  float brl_st;
  int brlp_enable;
  float brlp;
  float brlp_r;
  float brlp_g;
  float brlp_b;

  int hc_enable;
  float hc_r;
  float hc_r_rng;
  int hs_rgb_enable;
  float hs_r;
  float hs_r_rng;
  float hs_g;
  float hs_g_rng;
  float hs_b;
  float hs_b_rng;
  int hs_cmy_enable;
  float hs_c;
  float hs_c_rng;
  float hs_m;
  float hs_m_rng;
  float hs_y;
  float hs_y_rng;

  int clamp;
  int tn_su;
  int display_gamut;
  int eotf;
};

struct LookPresetValues {
  float tn_con, tn_sh, tn_toe, tn_off;
  int tn_hcon_enable;
  float tn_hcon, tn_hcon_pv, tn_hcon_st;
  int tn_lcon_enable;
  float tn_lcon, tn_lcon_w;
  int cwp;
  float cwp_lm;
  float rs_sa, rs_rw, rs_bw;
  int pt_enable;
  float pt_lml, pt_lml_r, pt_lml_g, pt_lml_b;
  float pt_lmh, pt_lmh_r, pt_lmh_b;
  int ptl_enable;
  float ptl_c, ptl_m, ptl_y;
  int ptm_enable;
  float ptm_low, ptm_low_rng, ptm_low_st;
  float ptm_high, ptm_high_rng, ptm_high_st;
  int brl_enable;
  float brl, brl_r, brl_g, brl_b, brl_rng, brl_st;
  int brlp_enable;
  float brlp, brlp_r, brlp_g, brlp_b;
  int hc_enable;
  float hc_r, hc_r_rng;
  int hs_rgb_enable;
  float hs_r, hs_r_rng, hs_g, hs_g_rng, hs_b, hs_b_rng;
  int hs_cmy_enable;
  float hs_c, hs_c_rng, hs_m, hs_m_rng, hs_y, hs_y_rng;
};

struct TonescalePresetValues {
  float tn_con, tn_sh, tn_toe, tn_off;
  int tn_hcon_enable;
  float tn_hcon, tn_hcon_pv, tn_hcon_st;
  int tn_lcon_enable;
  float tn_lcon, tn_lcon_w;
};

inline void setIntIfPresent(OFX::ImageEffect& fx, const char* name, int v);
inline void setBoolIfPresent(OFX::ImageEffect& fx, const char* name, bool v);
inline void setDoubleIfPresent(OFX::ImageEffect& fx, const char* name, double v);
inline void setStringIfPresent(OFX::ImageEffect& fx, const char* name, const std::string& v);

static constexpr std::array<const char*, 7> kLookPresetNames = {
  "Standard", "Arriba", "Sylvan", "Colorful", "Aery", "Dystopic", "Umbra"
};

static constexpr std::array<const char*, 7> kLookBaseTonescaleNames = {
  "Standard Tonescale", "Arriba Tonescale", "Sylvan Tonescale", "Colorful Tonescale",
  "Aery Tonescale", "Dystopic Tonescale", "Umbra Tonescale"
};

static constexpr std::array<const char*, 14> kTonescalePresetNames = {
  "USE LOOK PRESET",
  "Low Contrast", "Medium Contrast", "High Contrast",
  "Arriba Tonescale", "Sylvan Tonescale", "Colorful Tonescale",
  "Aery Tonescale", "Dystopic Tonescale", "Umbra Tonescale",
  "ACES-1.x", "ACES-2.0", "Marvelous Tonescape", "DaGrinchi ToneGroan"
};

static constexpr const char* kOpenDRTPortVersion = "1.1.0";
static constexpr int kOpenDRTBuildNumber = 13;

// Look preset table order must match kLookPresetNames and UI choice indices.
static constexpr std::array<LookPresetValues, 7> kLookPresets = {{
  {1.66f,0.5f,0.003f,0.005f,0,0.0f,1.0f,4.0f,0,0.0f,0.5f,2,0.25f,0.35f,0.25f,0.55f,1,0.25f,0.5f,0.0f,0.1f,0.25f,0.5f,0.0f,1,0.06f,0.08f,0.06f,1,0.4f,0.25f,0.5f,-0.8f,0.35f,0.4f,1,0.0f,-2.5f,-1.5f,-1.5f,0.5f,0.35f,1,-0.5f,-1.25f,-1.25f,-0.25f,1,1.0f,0.3f,1,0.6f,0.6f,0.35f,1.0f,0.66f,1.0f,1,0.25f,1.0f,0.0f,1.0f,0.0f,1.0f},
  {1.05f,0.5f,0.1f,0.01f,0,0.0f,1.0f,4.0f,1,1.5f,0.2f,2,0.25f,0.35f,0.25f,0.55f,1,0.25f,0.45f,0.0f,0.1f,0.25f,0.25f,0.0f,1,0.06f,0.08f,0.06f,1,1.0f,0.4f,0.5f,-0.8f,0.66f,0.6f,1,0.0f,-2.5f,-1.5f,-1.5f,0.5f,0.35f,1,0.0f,-1.7f,-2.0f,-0.5f,1,1.0f,0.3f,1,0.6f,0.8f,0.35f,1.0f,0.66f,1.0f,1,0.15f,1.0f,0.0f,1.0f,0.0f,1.0f},
  {1.6f,0.5f,0.01f,0.01f,0,0.0f,1.0f,4.0f,1,0.25f,0.75f,2,0.25f,0.25f,0.25f,0.55f,1,0.15f,0.5f,0.15f,0.1f,0.25f,0.15f,0.15f,1,0.05f,0.08f,0.05f,1,0.5f,0.5f,0.5f,-0.8f,0.5f,0.5f,1,-1.0f,-2.0f,-2.0f,0.0f,0.25f,0.25f,1,-1.0f,-0.5f,-0.25f,-0.25f,1,1.0f,0.4f,1,0.6f,1.15f,0.8f,1.25f,0.6f,1.0f,1,0.25f,0.25f,0.25f,0.5f,0.35f,0.5f},
  {1.5f,0.5f,0.003f,0.003f,0,0.0f,1.0f,4.0f,1,0.4f,0.5f,2,0.25f,0.35f,0.25f,0.55f,1,0.5f,1.0f,0.0f,0.5f,0.15f,0.15f,0.15f,1,0.05f,0.06f,0.05f,1,0.8f,0.5f,0.4f,-0.8f,0.4f,0.4f,1,0.0f,-1.25f,-1.25f,-0.25f,0.3f,0.5f,1,-0.5f,-1.25f,-1.25f,-0.5f,1,1.0f,0.4f,1,0.5f,0.8f,0.35f,1.0f,0.5f,1.0f,1,0.25f,1.0f,0.0f,1.0f,0.25f,1.0f},
  {1.15f,0.5f,0.04f,0.006f,0,0.0f,0.0f,0.5f,1,0.5f,2.0f,1,0.25f,0.25f,0.2f,0.5f,1,0.0f,0.5f,0.15f,0.1f,0.0f,0.1f,0.0f,1,0.05f,0.08f,0.05f,1,0.8f,0.35f,0.5f,-0.9f,0.5f,0.3f,1,-3.0f,0.0f,0.0f,1.0f,0.8f,0.15f,1,-1.0f,-1.0f,-1.0f,0.0f,1,0.5f,0.25f,1,0.6f,1.0f,0.35f,2.0f,0.5f,1.5f,1,0.35f,1.0f,0.25f,1.0f,0.35f,0.5f},
  {1.6f,0.5f,0.01f,0.008f,1,0.25f,0.0f,1.0f,1,1.0f,0.75f,3,0.25f,0.2f,0.25f,0.55f,1,0.15f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1,0.05f,0.08f,0.05f,1,0.25f,0.25f,0.8f,-0.8f,0.6f,0.25f,1,-2.0f,-2.0f,-2.0f,0.0f,0.35f,0.35f,1,0.0f,-1.0f,-1.0f,-1.0f,1,1.0f,0.25f,1,0.7f,1.33f,1.0f,2.0f,0.75f,2.0f,1,1.0f,0.5f,1.0f,1.0f,1.0f,0.765f},
  {1.8f,0.5f,0.001f,0.015f,0,0.0f,1.0f,4.0f,1,1.0f,1.0f,5,0.25f,0.35f,0.25f,0.55f,1,0.0f,0.5f,0.0f,0.15f,0.25f,0.25f,0.0f,1,0.05f,0.06f,0.05f,1,0.4f,0.35f,0.66f,-0.6f,0.45f,0.45f,1,-2.0f,-4.5f,-3.0f,-4.0f,0.35f,0.3f,1,0.0f,-2.0f,-1.0f,-0.5f,1,1.0f,0.35f,1,0.66f,1.0f,0.5f,2.0f,0.85f,2.0f,1,0.0f,1.0f,0.25f,1.0f,0.66f,0.66f}
}};

// Tonescale preset table excludes index 0 ("USE LOOK PRESET"), so callers offset by -1.
static constexpr std::array<TonescalePresetValues, 13> kTonescalePresets = {{
  {1.4f,0.5f,0.003f,0.005f,0,0.0f,1.0f,4.0f,0,0.0f,0.5f},
  {1.66f,0.5f,0.003f,0.005f,0,0.0f,1.0f,4.0f,0,0.0f,0.5f},
  {1.4f,0.5f,0.003f,0.005f,0,0.0f,1.0f,4.0f,1,1.0f,0.5f},
  {1.05f,0.5f,0.1f,0.01f,0,0.0f,1.0f,4.0f,1,1.5f,0.2f},
  {1.6f,0.5f,0.01f,0.01f,0,0.0f,1.0f,4.0f,1,0.25f,0.75f},
  {1.5f,0.5f,0.003f,0.003f,0,0.0f,1.0f,4.0f,1,0.4f,0.5f},
  {1.15f,0.5f,0.04f,0.006f,0,0.0f,0.0f,0.5f,1,0.5f,2.0f},
  {1.6f,0.5f,0.01f,0.008f,1,0.25f,0.0f,1.0f,1,1.0f,0.75f},
  {1.8f,0.5f,0.001f,0.015f,0,0.0f,1.0f,4.0f,1,1.0f,1.0f},
  {1.0f,0.35f,0.02f,0.0f,1,0.55f,0.0f,2.0f,1,1.13f,1.0f},
  {1.15f,0.5f,0.04f,0.0f,0,1.0f,1.0f,1.0f,0,1.0f,0.6f},
  {1.5f,0.5f,0.003f,0.01f,1,0.25f,0.0f,4.0f,1,1.0f,1.0f},
  {1.2f,0.5f,0.02f,0.0f,0,0.0f,1.0f,1.0f,0,0.0f,0.6f}
}};

// Label/metadata helpers for read-only UI state.
inline const char* currentPresetName(int presetIndex) {
  if (presetIndex < 0 || presetIndex >= static_cast<int>(kLookPresetNames.size())) {
    return "Standard";
  }
  return kLookPresetNames[static_cast<size_t>(presetIndex)];
}

inline std::string buildLabelText() {
  return std::string("v") + kOpenDRTPortVersion + " b" + std::to_string(kOpenDRTBuildNumber);
}

inline std::string presetLabelForClean(int presetIndex) {
  return std::string(currentPresetName(presetIndex)) + " | " + buildLabelText();
}

inline std::string presetLabelForCustom(int presetIndex) {
  return std::string("Custom (") + currentPresetName(presetIndex) + ") | " + buildLabelText();
}

inline const char* whitepointNameFromCwp(int cwp) {
  switch (cwp) {
    case 0: return "D93";
    case 1: return "D75";
    case 2: return "D65";
    case 3: return "D60";
    case 4: return "D55";
    case 5: return "D50";
    default: return "D65";
  }
}

inline std::string baseTonescaleLabelForLook(int presetIndex) {
  if (presetIndex < 0 || presetIndex >= static_cast<int>(kLookBaseTonescaleNames.size())) {
    return "Standard Tonescale";
  }
  return kLookBaseTonescaleNames[static_cast<size_t>(presetIndex)];
}

inline std::string baseWhitepointLabelForLook(int presetIndex) {
  const int idx = (presetIndex < 0 || presetIndex >= 7) ? 0 : presetIndex;
  const int cwp = kLookPresets[static_cast<size_t>(idx)].cwp;
  return std::string(whitepointNameFromCwp(cwp));
}

inline std::string effectiveTonescaleLabel(int lookPresetIndex, int tonescalePresetIndex, bool custom) {
  std::string base = (tonescalePresetIndex <= 0 || tonescalePresetIndex >= static_cast<int>(kTonescalePresetNames.size()))
      ? baseTonescaleLabelForLook(lookPresetIndex)
      : std::string(kTonescalePresetNames[static_cast<size_t>(tonescalePresetIndex)]);
  return custom ? (std::string("Custom (") + base + ")") : base;
}

inline std::string effectiveWhitepointLabel(int lookPresetIndex, int creativeWhitePresetIndex) {
  if (creativeWhitePresetIndex <= 0) {
    return baseWhitepointLabelForLook(lookPresetIndex);
  }
  return whitepointNameFromCwp(creativeWhitePresetIndex - 1);
}

inline const char* surroundNameFromIndex(int tn_su) {
  switch (tn_su) {
    case 0: return "Dark";
    case 1: return "Dim";
    case 2: return "Bright";
    default: return "Dim";
  }
}

// Resolved-params mutators used by resolveParams() and paramChanged() paths.
inline void applyLookPresetToResolved(OpenDRTParams& p, int lookPresetIndex) {
  const int idx = (lookPresetIndex < 0 || lookPresetIndex >= 7) ? 0 : lookPresetIndex;
  const LookPresetValues& s = kLookPresets[static_cast<size_t>(idx)];

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

inline void applyTonescalePresetToResolved(OpenDRTParams& p, int tonescalePresetIndex) {
  if (tonescalePresetIndex <= 0 || tonescalePresetIndex > 13) {
    return;
  }
  const TonescalePresetValues& t = kTonescalePresets[static_cast<size_t>(tonescalePresetIndex - 1)];
  p.tn_con = t.tn_con; p.tn_sh = t.tn_sh; p.tn_toe = t.tn_toe; p.tn_off = t.tn_off;
  p.tn_hcon_enable = t.tn_hcon_enable; p.tn_hcon = t.tn_hcon; p.tn_hcon_pv = t.tn_hcon_pv; p.tn_hcon_st = t.tn_hcon_st;
  p.tn_lcon_enable = t.tn_lcon_enable; p.tn_lcon = t.tn_lcon; p.tn_lcon_w = t.tn_lcon_w;
}

inline void applyDisplayEncodingPreset(OpenDRTParams& p, int preset) {
  switch (preset) {
    case 0: p.tn_su = 1; p.display_gamut = 0; p.eotf = 2; break;
    case 1: p.tn_su = 2; p.display_gamut = 0; p.eotf = 1; break;
    case 2: p.tn_su = 2; p.display_gamut = 1; p.eotf = 1; break;
    case 3: p.tn_su = 0; p.display_gamut = 3; p.eotf = 3; break;
    case 4: p.tn_su = 0; p.display_gamut = 4; p.eotf = 3; break;
    case 5: p.tn_su = 0; p.display_gamut = 5; p.eotf = 3; break;
    case 6: p.tn_su = 0; p.display_gamut = 2; p.eotf = 4; break;
    case 7: p.tn_su = 0; p.display_gamut = 2; p.eotf = 5; break;
    case 8: p.tn_su = 0; p.display_gamut = 1; p.eotf = 4; break;
    default: break;
  }
}

// OFX-param writers used when a preset is explicitly chosen in the UI.
inline void writeTonescalePresetToParams(int tonescalePresetIndex, OFX::ImageEffect& fx) {
  if (tonescalePresetIndex <= 0 || tonescalePresetIndex > 13) return;
  const TonescalePresetValues& t = kTonescalePresets[static_cast<size_t>(tonescalePresetIndex - 1)];
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

inline void writeCreativeWhitePresetToParams(int creativeWhitePresetIndex, OFX::ImageEffect& fx) {
  if (creativeWhitePresetIndex <= 0) return;
  setIntIfPresent(fx, "cwp", creativeWhitePresetIndex - 1);
}

inline void writeDisplayPresetToParams(int presetIndex, OFX::ImageEffect& fx) {
  // Display encoding presets are translated into explicit override params.
  OpenDRTParams p{};
  applyDisplayEncodingPreset(p, presetIndex);
  try {
    if (auto* q = fx.fetchChoiceParam("tn_su")) q->setValue(p.tn_su);
    if (auto* q = fx.fetchChoiceParam("display_gamut")) q->setValue(p.display_gamut);
    if (auto* q = fx.fetchChoiceParam("eotf")) q->setValue(p.eotf);
    setStringIfPresent(fx, "surroundLabel", surroundNameFromIndex(p.tn_su));
  } catch (...) {
  }
}

inline OpenDRTParams resolveParams(const OpenDRTRawValues& r) {
  // Resolve host/UI state into the flat kernel contract.
  OpenDRTParams p{};

  p.in_gamut = r.in_gamut;
  p.in_oetf = r.in_oetf;
  p.tn_Lp = r.tn_Lp;
  p.tn_gb = r.tn_gb;
  p.pt_hdr = r.pt_hdr;
  p.tn_Lg = r.tn_Lg;
  p.crv_enable = r.crv_enable;

  p.tn_con = r.tn_con;
  p.tn_sh = r.tn_sh;
  p.tn_toe = r.tn_toe;
  p.tn_off = r.tn_off;
  p.tn_hcon_enable = r.tn_hcon_enable;
  p.tn_hcon = r.tn_hcon;
  p.tn_hcon_pv = r.tn_hcon_pv;
  p.tn_hcon_st = r.tn_hcon_st;
  p.tn_lcon_enable = r.tn_lcon_enable;
  p.tn_lcon = r.tn_lcon;
  p.tn_lcon_w = r.tn_lcon_w;

  p.rs_sa = r.rs_sa;
  p.rs_rw = r.rs_rw;
  p.rs_bw = r.rs_bw;

  p.pt_enable = r.pt_enable;
  p.pt_lml = r.pt_lml;
  p.pt_lml_r = r.pt_lml_r;
  p.pt_lml_g = r.pt_lml_g;
  p.pt_lml_b = r.pt_lml_b;
  p.pt_lmh = r.pt_lmh;
  p.pt_lmh_r = r.pt_lmh_r;
  p.pt_lmh_b = r.pt_lmh_b;
  p.ptl_enable = r.ptl_enable;
  p.ptl_c = r.ptl_c;
  p.ptl_m = r.ptl_m;
  p.ptl_y = r.ptl_y;
  p.ptm_enable = r.ptm_enable;
  p.ptm_low = r.ptm_low;
  p.ptm_low_rng = r.ptm_low_rng;
  p.ptm_low_st = r.ptm_low_st;
  p.ptm_high = r.ptm_high;
  p.ptm_high_rng = r.ptm_high_rng;
  p.ptm_high_st = r.ptm_high_st;

  p.brl_enable = r.brl_enable;
  p.brl = r.brl;
  p.brl_r = r.brl_r;
  p.brl_g = r.brl_g;
  p.brl_b = r.brl_b;
  p.brl_rng = r.brl_rng;
  p.brl_st = r.brl_st;
  p.brlp_enable = r.brlp_enable;
  p.brlp = r.brlp;
  p.brlp_r = r.brlp_r;
  p.brlp_g = r.brlp_g;
  p.brlp_b = r.brlp_b;

  p.hc_enable = r.hc_enable;
  p.hc_r = r.hc_r;
  p.hc_r_rng = r.hc_r_rng;
  p.hs_rgb_enable = r.hs_rgb_enable;
  p.hs_r = r.hs_r;
  p.hs_r_rng = r.hs_r_rng;
  p.hs_g = r.hs_g;
  p.hs_g_rng = r.hs_g_rng;
  p.hs_b = r.hs_b;
  p.hs_b_rng = r.hs_b_rng;
  p.hs_cmy_enable = r.hs_cmy_enable;
  p.hs_c = r.hs_c;
  p.hs_c_rng = r.hs_c_rng;
  p.hs_m = r.hs_m;
  p.hs_m_rng = r.hs_m_rng;
  p.hs_y = r.hs_y;
  p.hs_y_rng = r.hs_y_rng;

  p.cwp = r.cwp;
  p.cwp_lm = r.creativeWhiteLimit;
  p.clamp = r.clamp;

  p.tn_su = r.tn_su;
  p.display_gamut = r.display_gamut;
  p.eotf = r.eotf;

  if (r.creativeWhitePreset > 0) {
    // Creative White preset overrides the base/look whitepoint index.
    p.cwp = r.creativeWhitePreset - 1;
  }

  return p;
}

inline void setIntIfPresent(OFX::ImageEffect& fx, const char* name, int v) {
  // Param lookup is host-dependent; missing params are ignored intentionally.
  try {
    if (auto* p = fx.fetchIntParam(name)) { p->setValue(v); }
  } catch (...) {
  }
}

inline void setBoolIfPresent(OFX::ImageEffect& fx, const char* name, bool v) {
  try {
    if (auto* p = fx.fetchBooleanParam(name)) { p->setValue(v ? 1 : 0); }
  } catch (...) {
  }
}

inline void setDoubleIfPresent(OFX::ImageEffect& fx, const char* name, double v) {
  try {
    if (auto* p = fx.fetchDoubleParam(name)) { p->setValue(v); }
  } catch (...) {
  }
}

inline void setStringIfPresent(OFX::ImageEffect& fx, const char* name, const std::string& v) {
  try {
    if (auto* p = fx.fetchStringParam(name)) { p->setValue(v); }
  } catch (...) {
  }
}

inline void writePresetToParams(int presetIndex, OFX::ImageEffect& fx) {
  // Preset selection is authoritative: overwrite all affected controls.
  const int idx = (presetIndex < 0 || presetIndex >= 7) ? 0 : presetIndex;
  const LookPresetValues& s = kLookPresets[static_cast<size_t>(idx)];

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

  setIntIfPresent(fx, "presetState", 0);
  setStringIfPresent(fx, "presetLabel", presetLabelForClean(idx));
  setStringIfPresent(fx, "baseTonescaleLabel", effectiveTonescaleLabel(idx, 0, false));
  setStringIfPresent(fx, "baseWhitepointLabel", effectiveWhitepointLabel(idx, 0));
}




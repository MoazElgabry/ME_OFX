#include <cmath>
#include <memory>
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
  }

  void render(const OFX::RenderArguments& args) override {
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

    std::vector<float> srcPixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);
    std::vector<float> dstPixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0.0f);

    for (int y = bounds.y1; y < bounds.y2; ++y) {
      for (int x = bounds.x1; x < bounds.x2; ++x) {
        float* sp = static_cast<float*>(src->getPixelAddress(x, y));
        const size_t i = (static_cast<size_t>(y - bounds.y1) * static_cast<size_t>(width) + static_cast<size_t>(x - bounds.x1)) * 4u;
        if (sp) {
          srcPixels[i + 0] = sp[0];
          srcPixels[i + 1] = sp[1];
          srcPixels[i + 2] = sp[2];
          srcPixels[i + 3] = sp[3];
        }
      }
    }

    OpenDRTRawValues raw = readRawValues(args.time);
    OpenDRTParams params = resolveParams(raw);
    OpenDRTProcessor processor(params);
    processor.render(srcPixels.data(), dstPixels.data(), width, height, true, false);

    for (int y = bounds.y1; y < bounds.y2; ++y) {
      for (int x = bounds.x1; x < bounds.x2; ++x) {
        float* dp = static_cast<float*>(dst->getPixelAddress(x, y));
        if (!dp) continue;
        const size_t i = (static_cast<size_t>(y - bounds.y1) * static_cast<size_t>(width) + static_cast<size_t>(x - bounds.x1)) * 4u;
        dp[0] = dstPixels[i + 0];
        dp[1] = dstPixels[i + 1];
        dp[2] = dstPixels[i + 2];
        dp[3] = dstPixels[i + 3];
      }
    }
  }

  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override {
    try {
      if (suppressParamChanged_) {
        return;
      }
      if (args.reason == OFX::eChangePluginEdit || args.reason == OFX::eChangeTime) {
        return;
      }

      if (paramName == "presetState" || paramName == "presetLabel" ||
          paramName == "baseTonescaleLabel" || paramName == "baseWhitepointLabel") {
        return;
      }
      if (paramName == "surroundLabel") {
        return;
      }

      if (paramName == "lookPreset") {
        int look = getChoice("lookPreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        writePresetToParams(look, *this);
        updatePresetStateFromCurrent(args.time);
        return;
      }

      if (paramName == "tonescalePreset") {
        const int look = getChoice("lookPreset", args.time, 0);
        const int tsPreset = getChoice("tonescalePreset", args.time, 0);
        FlagScope scope(suppressParamChanged_);
        writeTonescalePresetToParams(tsPreset, *this);
        setString("baseTonescaleLabel", effectiveTonescaleLabel(look, tsPreset, false));
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

      if (isAdvancedParam(paramName)) {
        FlagScope scope(suppressParamChanged_);
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

  bool almostEqual(float a, float b, float eps = 1e-6f) const {
    return std::fabs(a - b) <= eps;
  }

  bool isCurrentEqualToPresetBaseline(double time, bool* tonescaleCleanOut = nullptr) const {
    const int look = getChoice("lookPreset", time, 0);
    const int tsPreset = getChoice("tonescalePreset", time, 0);
    const int displayPreset = getChoice("displayEncodingPreset", time, 0);
    const int cwpPreset = getChoice("creativeWhitePreset", time, 0);

    OpenDRTParams expected{};
    applyLookPresetToResolved(expected, look);
    applyTonescalePresetToResolved(expected, tsPreset);
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
    bool tonescaleClean = true;
    const bool clean = isCurrentEqualToPresetBaseline(time, &tonescaleClean);
    setInt("presetState", clean ? 0 : 1);
    setString("presetLabel", clean ? presetLabelForClean(look) : presetLabelForCustom(look));
    setString("baseTonescaleLabel", effectiveTonescaleLabel(look, tsPreset, !tonescaleClean));
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
  bool suppressParamChanged_ = false;
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
    auto* pAdvanced = d.definePageParam("Advanced");
    auto* pOverlay = d.definePageParam("Overlay");

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
    auto* lookPreset = addChoice("lookPreset", "Look Preset", 0, {"Standard","Arriba","Sylvan","Colorful","Aery","Dystopic","Umbra"});
    auto* presetLabel = d.defineStringParam("presetLabel"); presetLabel->setLabel("Preset Label"); presetLabel->setDefault(presetLabelForClean(0)); presetLabel->setEnabled(false);
    auto* baseTsLabel = d.defineStringParam("baseTonescaleLabel"); baseTsLabel->setLabel("Base Tonescale"); baseTsLabel->setDefault(baseTonescaleLabelForLook(0)); baseTsLabel->setEnabled(false);
    auto* baseWpLabel = d.defineStringParam("baseWhitepointLabel"); baseWpLabel->setLabel("Base Whitepoint"); baseWpLabel->setDefault(baseWhitepointLabelForLook(0)); baseWpLabel->setEnabled(false);
    auto* surroundLabel = d.defineStringParam("surroundLabel"); surroundLabel->setLabel("Selected Surround"); surroundLabel->setDefault("Dim"); surroundLabel->setEnabled(false);
    auto* presetState = d.defineIntParam("presetState"); presetState->setIsSecret(true); presetState->setDefault(0);
    auto* cwpHidden = d.defineIntParam("cwp"); cwpHidden->setIsSecret(true); cwpHidden->setDefault(2);
    auto* tonescalePreset = addChoice("tonescalePreset", "Tonescale Preset", 0, {"USE LOOK PRESET","Low Contrast","Medium Contrast","High Contrast","Arriba Tonescale","Sylvan Tonescale","Colorful Tonescale","Aery Tonescale","Dystopic Tonescale","Umbra Tonescale","ACES-1.x","ACES-2.0","Marvelous Tonescape","DaGrinchi ToneGroan"});
    auto* cwpPreset = addChoice("creativeWhitePreset", "Creative White", 0, {"USE LOOK PRESET","D93","D75","D65","D60","D55","D50"});
    auto* cwpLm = addDouble("cwp_lm", "Creative White Limit", 0.25, 0.0, 1.0);
    pLook->addChild(*dep); pLook->addChild(*lookPreset); pLook->addChild(*presetLabel); pLook->addChild(*baseTsLabel); pLook->addChild(*baseWpLabel); pLook->addChild(*surroundLabel); pLook->addChild(*presetState); pLook->addChild(*cwpHidden); pLook->addChild(*tonescalePreset); pLook->addChild(*cwpPreset); pLook->addChild(*cwpLm);

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

    addAdvBool("pt_enable","Enable Purity Compress High",true,grpPurity);
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
    addAdvD("hs_r_rng","Hueshift R Range",0.6,0.0,2.0,grpHue);
    addAdvD("hs_g","Hueshift G",0.35,0.0,1.0,grpHue);
    addAdvD("hs_g_rng","Hueshift G Range",1.0,0.0,2.0,grpHue);
    addAdvD("hs_b","Hueshift B",0.66,0.0,1.0,grpHue);
    addAdvD("hs_b_rng","Hueshift B Range",1.0,0.0,4.0,grpHue);
    addAdvBool("hs_cmy_enable","Enable Hueshift CMY",true,grpHue);
    addAdvD("hs_c","Hueshift C",0.25,0.0,1.0,grpHue);
    addAdvD("hs_c_rng","Hueshift C Range",1.0,0.0,1.0,grpHue);
    addAdvD("hs_m","Hueshift M",0.0,0.0,1.0,grpHue);
    addAdvD("hs_m_rng","Hueshift M Range",1.0,0.0,1.0,grpHue);
    addAdvD("hs_y","Hueshift Y",0.0,0.0,1.0,grpHue);
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



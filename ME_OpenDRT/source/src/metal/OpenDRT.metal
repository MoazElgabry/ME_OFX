#include <metal_stdlib>
using namespace metal;
#include "../OpenDRTParams.h"
#define make_float3 float3
#define make_float2 float2


constant float SQRT3 = 1.7320508075688772f;
constant float PI = 3.1415926535897932f;

typedef struct {
  float3 x, y, z;
} drt_float3x3;

// Small custom 3x3 to keep math semantics aligned with CUDA port.
inline drt_float3x3 make_float3x3(float3 a, float3 b, float3 c) {
  drt_float3x3 d;
  d.x = a; d.y = b; d.z = c;
  return d;
}

inline drt_float3x3 identity() {
  return make_float3x3(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
}

inline float3 vdot(drt_float3x3 m, float3 v) {
  return make_float3(
      m.x.x * v.x + m.x.y * v.y + m.x.z * v.z,
      m.y.x * v.x + m.y.y * v.y + m.y.z * v.z,
      m.z.x * v.x + m.z.y * v.y + m.z.z * v.z);
}

#define matrix_ap0_to_xyz make_float3x3(make_float3(0.93863095f, -0.00574192f, 0.01756690f), make_float3(0.33809359f, 0.72721390f, -0.06530750f), make_float3(0.00072312f, 0.00081844f, 1.08751619f))
#define matrix_ap1_to_xyz make_float3x3(make_float3(0.65241872f, 0.12717993f, 0.17085728f), make_float3(0.26806406f, 0.67246448f, 0.05947146f), make_float3(-0.00546993f, 0.00518280f, 1.08934488f))
#define matrix_rec2020_to_xyz make_float3x3(make_float3(0.63695805f, 0.14461690f, 0.16888098f), make_float3(0.26270021f, 0.67799807f, 0.05930172f), make_float3(0.0f, 0.02807269f, 1.06098506f))
#define matrix_arriwg3_to_xyz make_float3x3(make_float3(0.63800762f, 0.21470386f, 0.09774445f), make_float3(0.29195378f, 0.82384104f, -0.11579482f), make_float3(0.00279828f, -0.06703424f, 1.15329371f))
#define matrix_arriwg4_to_xyz make_float3x3(make_float3(0.70485832f, 0.12976030f, 0.11583731f), make_float3(0.25452418f, 0.78147773f, -0.03600191f), make_float3(0.0f, 0.0f, 1.08905775f))
#define matrix_redwg_to_xyz make_float3x3(make_float3(0.73527525f, 0.06860941f, 0.14657127f), make_float3(0.28669410f, 0.84297913f, -0.12967323f), make_float3(-0.07968086f, -0.34734322f, 1.51608182f))
#define matrix_sonysgamut3_to_xyz make_float3x3(make_float3(0.70648271f, 0.12880105f, 0.11517216f), make_float3(0.27097967f, 0.78660641f, -0.05758608f), make_float3(-0.00967785f, 0.00460004f, 1.09413556f))
#define matrix_sonysgamut3cine_to_xyz make_float3x3(make_float3(0.59908392f, 0.24892552f, 0.10244649f), make_float3(0.21507582f, 0.88506850f, -0.10014432f), make_float3(-0.03206585f, -0.02765839f, 1.14878199f))
#define matrix_vgamut_to_xyz make_float3x3(make_float3(0.67964447f, 0.15221141f, 0.11860004f), make_float3(0.26068555f, 0.77489446f, -0.03558001f), make_float3(-0.00931020f, -0.00461247f, 1.10298042f))
#define matrix_egamut_to_xyz make_float3x3(make_float3(0.70539685f, 0.16404133f, 0.08101775f), make_float3(0.28013072f, 0.82020664f, -0.10033737f), make_float3(-0.10378151f, -0.07290726f, 1.26574652f))
#define matrix_egamut2_to_xyz make_float3x3(make_float3(0.73647770f, 0.13073965f, 0.08323858f), make_float3(0.27506998f, 0.82801779f, -0.10308777f), make_float3(-0.12422515f, -0.08715977f, 1.30044267f))
#define matrix_davinciwg_to_xyz make_float3x3(make_float3(0.70062239f, 0.14877482f, 0.10105872f), make_float3(0.27411851f, 0.87363190f, -0.14775041f), make_float3(-0.09896291f, -0.13789533f, 1.32591599f))
#define matrix_p3d65_to_xyz make_float3x3(make_float3(0.48657095f, 0.26566769f, 0.19821729f), make_float3(0.22897456f, 0.69173852f, 0.07928691f), make_float3(0.0f, 0.04511338f, 1.04394437f))
#define matrix_xyz_to_p3d65 make_float3x3(make_float3(2.49349691f, -0.93138362f, -0.40271078f), make_float3(-0.82948897f, 1.76266406f, 0.02362469f), make_float3(0.03584583f, -0.07617239f, 0.95688452f))
#define matrix_rec709_to_xyz make_float3x3(make_float3(0.41239080f, 0.35758434f, 0.18048079f), make_float3(0.21263901f, 0.71516868f, 0.07219232f), make_float3(0.01933082f, 0.11919478f, 0.95053215f))
#define matrix_xyz_to_rec709 make_float3x3(make_float3(3.24096994f, -1.53738318f, -0.49861076f), make_float3(-0.96924364f, 1.87596750f, 0.04155506f), make_float3(0.05563008f, -0.20397696f, 1.05697151f))
#define matrix_p3_to_rec2020 make_float3x3(make_float3(0.75383303f, 0.19859737f, 0.04756960f), make_float3(0.04574385f, 0.94177722f, 0.01247893f), make_float3(-0.00121034f, 0.01760172f, 0.98360862f))

#define matrix_cat_dci_to_d93 make_float3x3(make_float3(0.96568501f, 0.00183745f, 0.09129673f), make_float3(0.00051457f, 0.96516675f, 0.03601465f), make_float3(0.00154250f, 0.00702652f, 1.47287476f))
#define matrix_cat_dci_to_d75 make_float3x3(make_float3(0.99012077f, 0.01513895f, 0.05110477f), make_float3(0.01021972f, 0.97171813f, 0.02005366f), make_float3(0.00074307f, 0.00421763f, 1.27959657f))
#define matrix_cat_dci_to_d65 make_float3x3(make_float3(1.00951600f, 0.02696754f, 0.02136208f), make_float3(0.01879910f, 0.97533035f, 0.00822733f), make_float3(0.00013454f, 0.00217903f, 1.13866365f))
#define matrix_cat_dci_to_d60 make_float3x3(make_float3(1.02159524f, 0.03484868f, 0.00371252f), make_float3(0.02449688f, 0.97693723f, 0.00120302f), make_float3(-0.00023392f, 0.00098669f, 1.05594265f))
#define matrix_cat_dci_to_d55 make_float3x3(make_float3(1.03594577f, 0.04509376f, -0.01575738f), make_float3(0.03187407f, 0.97774452f, -0.00655745f), make_float3(-0.00065361f, -0.00029737f, 0.96632779f))
#define matrix_cat_dci_to_d50 make_float3x3(make_float3(1.05306876f, 0.05812973f, -0.03761008f), make_float3(0.04123594f, 0.97769368f, -0.01527922f), make_float3(-0.00113778f, -0.00170759f, 0.86736834f))
#define matrix_cat_d65_to_d93 make_float3x3(make_float3(0.95703423f, -0.02471715f, 0.06240286f), make_float3(-0.01792970f, 0.99001986f, 0.02481195f), make_float3(0.00127589f, 0.00427919f, 1.29345715f))
#define matrix_cat_d65_to_d75 make_float3x3(make_float3(0.98100108f, -0.01166193f, 0.02656141f), make_float3(-0.00843488f, 0.99650609f, 0.01056965f), make_float3(0.00055281f, 0.00179841f, 1.12374723f))
#define matrix_cat_d65_to_d60 make_float3x3(make_float3(1.01182246f, 0.00778879f, -0.01577830f), make_float3(0.00561683f, 1.00150645f, -0.00628518f), make_float3(-0.00033574f, -0.00105095f, 0.92736667f))
#define matrix_cat_d65_to_d55 make_float3x3(make_float3(1.02585089f, 0.01794398f, -0.03321378f), make_float3(0.01291339f, 1.00214779f, -0.01324210f), make_float3(-0.00071994f, -0.00218107f, 0.84868014f))
#define matrix_cat_d65_to_d50 make_float3x3(make_float3(1.04257405f, 0.03089118f, -0.05281262f), make_float3(0.02219354f, 1.00185668f, -0.02107376f), make_float3(-0.00116488f, -0.00342053f, 0.76178908f))
#define matrix_cat_d65_to_dci make_float3x3(make_float3(0.99108559f, -0.02736229f, -0.01839566f), make_float3(-0.01910219f, 1.02583778f, -0.00705373f), make_float3(-0.00008055f, -0.00195989f, 0.87823844f))
#define matrix_cat_d60_to_d93 make_float3x3(make_float3(0.94605690f, -0.03195030f, 0.08317015f), make_float3(-0.02319797f, 0.98874581f, 0.03306175f), make_float3(0.00169203f, 0.00572329f, 1.39483106f))
#define matrix_cat_d60_to_d75 make_float3x3(make_float3(0.96965998f, -0.01913831f, 0.04500996f), make_float3(-0.01385458f, 0.99513382f, 0.01790623f), make_float3(0.00093145f, 0.00306008f, 1.21179807f))
#define matrix_cat_d60_to_d65 make_float3x3(make_float3(0.98836392f, -0.00766910f, 0.01676416f), make_float3(-0.00554096f, 0.99854612f, 0.00667332f), make_float3(0.00035154f, 0.00112884f, 1.07833576f))
#define matrix_cat_d60_to_d55 make_float3x3(make_float3(1.01380289f, 0.01001315f, -0.01849835f), make_float3(0.00720565f, 1.00057685f, -0.00737530f), make_float3(-0.00040113f, -0.00121435f, 0.91513568f))
#define matrix_cat_d60_to_d50 make_float3x3(make_float3(1.03025270f, 0.02279105f, -0.03926569f), make_float3(0.01637665f, 1.00020599f, -0.01566682f), make_float3(-0.00086458f, -0.00254668f, 0.82142204f))

inline float sdivf(float a, float b) { return (b == 0.0f) ? 0.0f : a / b; }
inline float3 sdivf3f(float3 a, float b) { return make_float3(sdivf(a.x, b), sdivf(a.y, b), sdivf(a.z, b)); }
inline float spowf(float a, float b) { return (a <= 0.0f) ? a : pow(a, b); }
inline float3 spowf3(float3 a, float b) { return make_float3(spowf(a.x, b), spowf(a.y, b), spowf(a.z, b)); }
inline float hypotf2(float2 v) { return sqrt(fmax(0.0f, v.x * v.x + v.y * v.y)); }
inline float hypotf3(float3 v) { return sqrt(fmax(0.0f, v.x * v.x + v.y * v.y + v.z * v.z)); }
inline float3 clampf3(float3 a, float mn, float mx) { return make_float3(fmin(fmax(a.x, mn), mx), fmin(fmax(a.y, mn), mx), fmin(fmax(a.z, mn), mx)); }
inline float3 clampminf3(float3 a, float mn) { return make_float3(fmax(a.x, mn), fmax(a.y, mn), fmax(a.z, mn)); }

inline float oetf_davinci_intermediate(float x) { return x <= 0.02740668f ? x / 10.44426855f : exp2(x / 0.07329248f - 7.0f) - 0.0075f; }
inline float oetf_filmlight_tlog(float x) { return x < 0.075f ? (x - 0.075f) / 16.184376489665897f : exp((x - 0.5520126568606655f) / 0.09232902596577353f) - 0.0057048244042473785f; }
inline float oetf_acescct(float x) { return x <= 0.155251141552511f ? (x - 0.0729055341958355f) / 10.5402377416545f : exp2(x * 17.52f - 9.72f); }
inline float exp10_compat(float x) { return exp2(x * 3.3219280948873626f); }
inline float oetf_arri_logc3(float x) { return x < 5.367655f * 0.010591f + 0.092809f ? (x - 0.092809f) / 5.367655f : (exp10_compat((x - 0.385537f) / 0.247190f) - 0.052272f) / 5.555556f; }
inline float oetf_arri_logc4(float x) { return x < -0.7774983977293537f ? x * 0.3033266726886969f - 0.7774983977293537f : (exp2(14.0f * (x - 0.09286412512218964f) / 0.9071358748778103f + 6.0f) - 64.0f) / 2231.8263090676883f; }
inline float oetf_red_log3g10(float x) { return x < 0.0f ? (x / 15.1927f) - 0.01f : (exp10_compat(x / 0.224282f) - 1.0f) / 155.975327f - 0.01f; }
inline float oetf_panasonic_vlog(float x) { return x < 0.181f ? (x - 0.125f) / 5.6f : exp10_compat((x - 0.598206f) / 0.241514f) - 0.00873f; }
inline float oetf_sony_slog3(float x) { return x < 171.2102946929f / 1023.0f ? (x * 1023.0f - 95.0f) * 0.01125f / (171.2102946929f - 95.0f) : (exp10_compat(((x * 1023.0f - 420.0f) / 261.5f)) * (0.18f + 0.01f) - 0.01f); }
inline float oetf_fujifilm_flog2(float x) { return x < 0.100686685370811f ? (x - 0.092864f) / 8.799461f : (exp10_compat(((x - 0.384316f) / 0.245281f)) / 5.555556f - 0.064829f / 5.555556f); }

inline float3 linearize(float3 rgb, int tf) {
  if (tf == 0) return rgb;
  if (tf == 1) { rgb.x = oetf_davinci_intermediate(rgb.x); rgb.y = oetf_davinci_intermediate(rgb.y); rgb.z = oetf_davinci_intermediate(rgb.z); }
  else if (tf == 2) { rgb.x = oetf_filmlight_tlog(rgb.x); rgb.y = oetf_filmlight_tlog(rgb.y); rgb.z = oetf_filmlight_tlog(rgb.z); }
  else if (tf == 3) { rgb.x = oetf_acescct(rgb.x); rgb.y = oetf_acescct(rgb.y); rgb.z = oetf_acescct(rgb.z); }
  else if (tf == 4) { rgb.x = oetf_arri_logc3(rgb.x); rgb.y = oetf_arri_logc3(rgb.y); rgb.z = oetf_arri_logc3(rgb.z); }
  else if (tf == 5) { rgb.x = oetf_arri_logc4(rgb.x); rgb.y = oetf_arri_logc4(rgb.y); rgb.z = oetf_arri_logc4(rgb.z); }
  else if (tf == 6) { rgb.x = oetf_red_log3g10(rgb.x); rgb.y = oetf_red_log3g10(rgb.y); rgb.z = oetf_red_log3g10(rgb.z); }
  else if (tf == 7) { rgb.x = oetf_panasonic_vlog(rgb.x); rgb.y = oetf_panasonic_vlog(rgb.y); rgb.z = oetf_panasonic_vlog(rgb.z); }
  else if (tf == 8) { rgb.x = oetf_sony_slog3(rgb.x); rgb.y = oetf_sony_slog3(rgb.y); rgb.z = oetf_sony_slog3(rgb.z); }
  else if (tf == 9) { rgb.x = oetf_fujifilm_flog2(rgb.x); rgb.y = oetf_fujifilm_flog2(rgb.y); rgb.z = oetf_fujifilm_flog2(rgb.z); }
  return rgb;
}

inline float3 eotf_hlg(float3 rgb, int inverse) {
  if (inverse == 1) {
    float Yd = 0.2627f * rgb.x + 0.6780f * rgb.y + 0.0593f * rgb.z;
    rgb = rgb * spowf(Yd, (1.0f - 1.2f) / 1.2f);
    rgb.x = rgb.x <= 1.0f / 12.0f ? sqrt(3.0f * rgb.x) : 0.17883277f * log(12.0f * rgb.x - 0.28466892f) + 0.55991073f;
    rgb.y = rgb.y <= 1.0f / 12.0f ? sqrt(3.0f * rgb.y) : 0.17883277f * log(12.0f * rgb.y - 0.28466892f) + 0.55991073f;
    rgb.z = rgb.z <= 1.0f / 12.0f ? sqrt(3.0f * rgb.z) : 0.17883277f * log(12.0f * rgb.z - 0.28466892f) + 0.55991073f;
  } else {
    rgb.x = rgb.x <= 0.5f ? rgb.x * rgb.x / 3.0f : (exp((rgb.x - 0.55991073f) / 0.17883277f) + 0.28466892f) / 12.0f;
    rgb.y = rgb.y <= 0.5f ? rgb.y * rgb.y / 3.0f : (exp((rgb.y - 0.55991073f) / 0.17883277f) + 0.28466892f) / 12.0f;
    rgb.z = rgb.z <= 0.5f ? rgb.z * rgb.z / 3.0f : (exp((rgb.z - 0.55991073f) / 0.17883277f) + 0.28466892f) / 12.0f;
    float Ys = 0.2627f * rgb.x + 0.6780f * rgb.y + 0.0593f * rgb.z;
    rgb = rgb * spowf(Ys, 0.2f);
  }
  return rgb;
}

inline float3 eotf_pq(float3 rgb, int inverse) {
  const float m1 = 2610.0f / 16384.0f;
  const float m2 = 2523.0f / 32.0f;
  const float c1 = 107.0f / 128.0f;
  const float c2 = 2413.0f / 128.0f;
  const float c3 = 2392.0f / 128.0f;
  if (inverse == 1) {
    rgb = spowf3(rgb, m1);
    rgb = spowf3((c1 + c2 * rgb) / (1.0f + c3 * rgb), m2);
  } else {
    rgb = spowf3(rgb, 1.0f / m2);
    rgb = spowf3((rgb - c1) / (c2 - c3 * rgb), 1.0f / m1);
  }
  return rgb;
}

inline float compress_hyperbolic_power(float x, float s, float p) { return spowf(x / (x + s), p); }
inline float compress_toe_quadratic(float x, float toe, int inv) { if (toe == 0.0f) return x; return inv == 0 ? spowf(x, 2.0f) / (x + toe) : (x + sqrt(x * (4.0f * toe + x))) / 2.0f; }
inline float compress_toe_cubic(float x, float m, float w, int inv) {
  if (m == 1.0f) return x;
  float x2 = x * x;
  if (inv == 0) return x * (x2 + m * w) / (x2 + w);
  float p0 = x2 - 3.0f * m * w;
  float p1 = 2.0f * x2 + 27.0f * w - 9.0f * m * w;
  float radicand = fmax(0.0f, x2 * p1 * p1 - 4.0f * p0 * p0 * p0);
  float base = sqrt(radicand) / 2.0f + x * p1 / 2.0f;
  float p2 = base <= 0.0f ? 0.0f : pow(base, 1.0f / 3.0f);
  if (p2 == 0.0f) return x;
  return p0 / (3.0f * p2) + p2 / 3.0f + x / 3.0f;
}
inline float softplus(float x, float s) { if (x > 10.0f * s || s < 1e-4f) return x; return s * log(fmax(0.0f, 1.0f + exp(x / s))); }
inline float gauss_window(float x, float w) { return exp(-x * x / w); }
inline float2 opponent(float3 rgb) { return make_float2(rgb.x - rgb.z, rgb.y - (rgb.x + rgb.z) / 2.0f); }
inline float hue_offset(float h, float o) { return fmod(h - o + PI, 2.0f * PI) - PI; }
inline float contrast_high(float x, float p, float pv, float pv_lx, int inv) {
  const float x0 = 0.18f * pow(2.0f, pv);
  if (x < x0 || p == 1.0f) return x;
  const float o = x0 - x0 / p;
  const float s0 = pow(x0, 1.0f - p) / p;
  const float x1 = x0 * pow(2.0f, pv_lx);
  const float k1 = p * s0 * pow(x1, p) / x1;
  const float y1 = s0 * pow(x1, p) + o;
  if (inv == 1) {
    if (x > y1) return (x - y1) / k1 + x1;
    float base = (x - o) / s0;
    return base <= 0.0f ? 0.0f : pow(base, 1.0f / p);
  }
  return x > x1 ? k1 * (x - x1) + y1 : s0 * pow(x, p) + o;
}

inline float3 display_gamut_whitepoint(float3 rgb, float tsn, float cwp_lm, int display_gamut, int cwp) {
  rgb = vdot(matrix_p3d65_to_xyz, rgb);
  float3 cwp_neutral = rgb;
  float cwp_f = pow(tsn, 2.0f * cwp_lm);
  if (display_gamut < 3) {
    if (cwp == 0) rgb = vdot(matrix_cat_d65_to_d93, rgb);
    else if (cwp == 1) rgb = vdot(matrix_cat_d65_to_d75, rgb);
    else if (cwp == 3) rgb = vdot(matrix_cat_d65_to_d60, rgb);
    else if (cwp == 4) rgb = vdot(matrix_cat_d65_to_d55, rgb);
    else if (cwp == 5) rgb = vdot(matrix_cat_d65_to_d50, rgb);
  } else if (display_gamut == 3) {
    if (cwp == 0) rgb = vdot(matrix_cat_d60_to_d93, rgb);
    else if (cwp == 1) rgb = vdot(matrix_cat_d60_to_d75, rgb);
    else if (cwp == 2) rgb = vdot(matrix_cat_d60_to_d65, rgb);
    else if (cwp == 4) rgb = vdot(matrix_cat_d60_to_d55, rgb);
    else if (cwp == 5) rgb = vdot(matrix_cat_d60_to_d50, rgb);
  } else {
    cwp_neutral = vdot(matrix_cat_dci_to_d65, rgb);
    if (cwp == 0) rgb = vdot(matrix_cat_dci_to_d93, rgb);
    else if (cwp == 1) rgb = vdot(matrix_cat_dci_to_d75, rgb);
    else if (cwp == 2) rgb = cwp_neutral;
    else if (cwp == 3) rgb = vdot(matrix_cat_dci_to_d60, rgb);
    else if (cwp == 4) rgb = vdot(matrix_cat_dci_to_d55, rgb);
    else if (cwp == 5) rgb = vdot(matrix_cat_dci_to_d50, rgb);
  }
  rgb = rgb * cwp_f + cwp_neutral * (1.0f - cwp_f);
  if (display_gamut == 0) rgb = vdot(matrix_xyz_to_rec709, rgb);
  else if (display_gamut == 5) rgb = vdot(matrix_cat_d65_to_dci, rgb);
  else rgb = vdot(matrix_xyz_to_p3d65, rgb);

  float cwp_norm = 1.0f;
  if (display_gamut == 0) {
    if (cwp == 0) cwp_norm = 0.74419270f; else if (cwp == 1) cwp_norm = 0.87347083f; else if (cwp == 3) cwp_norm = 0.95593699f; else if (cwp == 4) cwp_norm = 0.90567133f; else if (cwp == 5) cwp_norm = 0.85000439f;
  } else if (display_gamut == 1 || display_gamut == 2) {
    if (cwp == 0) cwp_norm = 0.76268706f; else if (cwp == 1) cwp_norm = 0.88405408f; else if (cwp == 3) cwp_norm = 0.96432019f; else if (cwp == 4) cwp_norm = 0.92307652f; else if (cwp == 5) cwp_norm = 0.87657284f;
  } else if (display_gamut == 3) {
    if (cwp == 0) cwp_norm = 0.70495632f; else if (cwp == 1) cwp_norm = 0.81671571f; else if (cwp == 2) cwp_norm = 0.92338219f; else if (cwp == 4) cwp_norm = 0.95613850f; else if (cwp == 5) cwp_norm = 0.90680145f;
  } else if (display_gamut == 4) {
    if (cwp == 0) cwp_norm = 0.66533614f; else if (cwp == 1) cwp_norm = 0.77039713f; else if (cwp == 2) cwp_norm = 0.87057234f; else if (cwp == 3) cwp_norm = 0.89135455f; else if (cwp == 4) cwp_norm = 0.85532783f; else if (cwp == 5) cwp_norm = 0.81456644f;
  } else if (display_gamut == 5) {
    if (cwp == 0) cwp_norm = 0.70714278f; else if (cwp == 1) cwp_norm = 0.81556108f; else cwp_norm = 0.91655528f;
  }
  rgb *= cwp_norm * cwp_f + 1.0f - cwp_f;
  return rgb;
}

// Core OpenDRT per-pixel transform. Input/output are display-agnostic RGB ratios + tonescale.
inline float3 openDRTTransform(
    int width, int height, int x, int y, float3 rgb, constant OpenDRTParams& p, constant OpenDRTDerivedParams& d) {
  int in_gamut = p.in_gamut, in_oetf = p.in_oetf, crv_enable = p.crv_enable, tn_hcon_enable = p.tn_hcon_enable, tn_lcon_enable = p.tn_lcon_enable;
  int pt_enable = p.pt_enable, ptl_enable = p.ptl_enable, ptm_enable = p.ptm_enable, brl_enable = p.brl_enable, brlp_enable = p.brlp_enable, hc_enable = p.hc_enable, hs_rgb_enable = p.hs_rgb_enable, hs_cmy_enable = p.hs_cmy_enable;
  int cwp = p.cwp, tn_su = p.tn_su, clamp = p.clamp, display_gamut = p.display_gamut, eotf = p.eotf;
  float tn_Lp = p.tn_Lp, tn_gb = p.tn_gb, pt_hdr = p.pt_hdr, tn_Lg = p.tn_Lg, tn_con = p.tn_con, tn_sh = p.tn_sh, tn_toe = p.tn_toe, tn_off = p.tn_off;
  float tn_hcon = p.tn_hcon, tn_hcon_pv = p.tn_hcon_pv, tn_hcon_st = p.tn_hcon_st, tn_lcon = p.tn_lcon, tn_lcon_w = p.tn_lcon_w;
  float rs_sa = p.rs_sa, rs_rw = p.rs_rw, rs_bw = p.rs_bw;
  float pt_lml = p.pt_lml, pt_lml_r = p.pt_lml_r, pt_lml_g = p.pt_lml_g, pt_lml_b = p.pt_lml_b, pt_lmh = p.pt_lmh, pt_lmh_r = p.pt_lmh_r, pt_lmh_b = p.pt_lmh_b;
  float ptl_c = p.ptl_c, ptl_m = p.ptl_m, ptl_y = p.ptl_y, ptm_low = p.ptm_low, ptm_low_rng = p.ptm_low_rng, ptm_low_st = p.ptm_low_st, ptm_high = p.ptm_high, ptm_high_rng = p.ptm_high_rng, ptm_high_st = p.ptm_high_st;
  float brl = p.brl, brl_r = p.brl_r, brl_g = p.brl_g, brl_b = p.brl_b, brl_rng = p.brl_rng, brl_st = p.brl_st;
  float brlp = p.brlp, brlp_r = p.brlp_r, brlp_g = p.brlp_g, brlp_b = p.brlp_b;
  float hc_r = p.hc_r, hc_r_rng = p.hc_r_rng, hs_r = p.hs_r, hs_r_rng = p.hs_r_rng, hs_g = p.hs_g, hs_g_rng = p.hs_g_rng, hs_b = p.hs_b, hs_b_rng = p.hs_b_rng;
  float hs_c = p.hs_c, hs_c_rng = p.hs_c_rng, hs_m = p.hs_m, hs_m_rng = p.hs_m_rng, hs_y = p.hs_y, hs_y_rng = p.hs_y_rng, cwp_lm = p.cwp_lm;

  drt_float3x3 in_to_xyz;
  if (in_gamut == 0) in_to_xyz = identity();
  else if (in_gamut == 1) in_to_xyz = matrix_ap0_to_xyz;
  else if (in_gamut == 2) in_to_xyz = matrix_ap1_to_xyz;
  else if (in_gamut == 3) in_to_xyz = matrix_p3d65_to_xyz;
  else if (in_gamut == 4) in_to_xyz = matrix_rec2020_to_xyz;
  else if (in_gamut == 5) in_to_xyz = matrix_rec709_to_xyz;
  else if (in_gamut == 6) in_to_xyz = matrix_arriwg3_to_xyz;
  else if (in_gamut == 7) in_to_xyz = matrix_arriwg4_to_xyz;
  else if (in_gamut == 8) in_to_xyz = matrix_redwg_to_xyz;
  else if (in_gamut == 9) in_to_xyz = matrix_sonysgamut3_to_xyz;
  else if (in_gamut == 10) in_to_xyz = matrix_sonysgamut3cine_to_xyz;
  else if (in_gamut == 11) in_to_xyz = matrix_vgamut_to_xyz;
  else if (in_gamut == 12) in_to_xyz = matrix_egamut_to_xyz;
  else if (in_gamut == 13) in_to_xyz = matrix_egamut2_to_xyz;
  else in_to_xyz = matrix_davinciwg_to_xyz;

  float crv_tsn = 0.0f;
  float2 pos = make_float2((float)x, (float)y), res = make_float2((float)width, (float)height);
  if (crv_enable == 1) crv_tsn = oetf_filmlight_tlog(pos.x / res.x);
  rgb = linearize(rgb, in_oetf);

  float ts_x1, ts_y1, ts_x0, ts_y0, ts_s0, ts_p, ts_s10, ts_m1, ts_m2, ts_s, ts_dsc, pt_cmp_Lf, s_Lp100, ts_s1;
  if (d.enabled != 0) {
    ts_x1 = d.ts_x1;
    ts_y1 = d.ts_y1;
    ts_x0 = d.ts_x0;
    ts_y0 = d.ts_y0;
    ts_s0 = d.ts_s0;
    ts_p = d.ts_p;
    ts_s10 = d.ts_s10;
    ts_m1 = d.ts_m1;
    ts_m2 = d.ts_m2;
    ts_s = d.ts_s;
    ts_dsc = d.ts_dsc;
    pt_cmp_Lf = d.pt_cmp_Lf;
    s_Lp100 = d.s_Lp100;
    ts_s1 = d.ts_s1;
  } else {
    ts_x1 = pow(2.0f, 6.0f * tn_sh + 4.0f);
    ts_y1 = tn_Lp / 100.0f;
    ts_x0 = 0.18f + tn_off;
    ts_y0 = tn_Lg / 100.0f * (1.0f + tn_gb * log2(ts_y1));
    ts_s0 = compress_toe_quadratic(ts_y0, tn_toe, 1);
    ts_p = tn_con / (1.0f + (float)tn_su * 0.05f);
    ts_s10 = ts_x0 * (pow(ts_s0, -1.0f / tn_con) - 1.0f);
    ts_m1 = ts_y1 / pow(ts_x1 / (ts_x1 + ts_s10), tn_con);
    ts_m2 = compress_toe_quadratic(ts_m1, tn_toe, 1);
    ts_s = ts_x0 * (pow(ts_s0 / ts_m2, -1.0f / tn_con) - 1.0f);
    ts_dsc = eotf == 4 ? 0.01f : eotf == 5 ? 0.1f : 100.0f / tn_Lp;
    pt_cmp_Lf = pt_hdr * fmin(1.0f, (tn_Lp - 100.0f) / 900.0f);
    s_Lp100 = ts_x0 * (pow((tn_Lg / 100.0f), -1.0f / tn_con) - 1.0f);
    ts_s1 = ts_s * pt_cmp_Lf + s_Lp100 * (1.0f - pt_cmp_Lf);
  }

  rgb = vdot(matrix_xyz_to_p3d65, vdot(in_to_xyz, rgb));
  float3 rs_w = make_float3(rs_rw, 1.0f - rs_rw - rs_bw, rs_bw);
  float sat_L = rgb.x * rs_w.x + rgb.y * rs_w.y + rgb.z * rs_w.z;
  rgb = sat_L * rs_sa + rgb * (1.0f - rs_sa);
  rgb += tn_off;
  if (crv_enable == 1) crv_tsn += tn_off;

  float tsn = hypotf3(rgb) / SQRT3;
  rgb = sdivf3f(rgb, tsn);
  float2 opp = opponent(rgb);
  float ach_d = 1.25f * compress_toe_quadratic(hypotf2(opp) / 2.0f, 0.25f, 0);
  float hue = fmod(atan2(opp.x, opp.y) + PI + 1.10714931f, 2.0f * PI);

  float3 ha_rgb = make_float3(gauss_window(hue_offset(hue, 0.1f), 0.66f), gauss_window(hue_offset(hue, 4.3f), 0.66f), gauss_window(hue_offset(hue, 2.3f), 0.66f));
  float3 ha_rgb_hs = make_float3(gauss_window(hue_offset(hue, -0.4f), 0.66f), ha_rgb.y, gauss_window(hue_offset(hue, 2.5f), 0.66f));
  float3 ha_cmy = make_float3(gauss_window(hue_offset(hue, 3.3f), 0.5f), gauss_window(hue_offset(hue, 1.3f), 0.5f), gauss_window(hue_offset(hue, -1.15f), 0.5f));

  if (brl_enable) {
    float brl_tsf = pow(tsn / (tsn + 1.0f), 1.0f - brl_rng);
    float brl_exf = (brl + brl_r * ha_rgb.x + brl_g * ha_rgb.y + brl_b * ha_rgb.z) * pow(ach_d, 1.0f / brl_st);
    float brl_ex = pow(2.0f, brl_exf * (brl_exf < 0.0f ? brl_tsf : 1.0f - brl_tsf));
    tsn *= brl_ex;
  }

  if (tn_lcon_enable) {
    float lcon_m = pow(2.0f, -tn_lcon), lcon_w = (tn_lcon_w / 4.0f); lcon_w *= lcon_w;
    const float lcon_cnst_sc = compress_toe_cubic(ts_x0, lcon_m, lcon_w, 1) / ts_x0;
    tsn = compress_toe_cubic(tsn * lcon_cnst_sc, lcon_m, lcon_w, 0);
    if (crv_enable == 1) crv_tsn = compress_toe_cubic(crv_tsn * lcon_cnst_sc, lcon_m, lcon_w, 0);
  }

  if (tn_hcon_enable) {
    float hcon_p = pow(2.0f, tn_hcon);
    tsn = contrast_high(tsn, hcon_p, tn_hcon_pv, tn_hcon_st, 0);
    if (crv_enable == 1) crv_tsn = contrast_high(crv_tsn, hcon_p, tn_hcon_pv, tn_hcon_st, 0);
  }

  float tsn_pt = compress_hyperbolic_power(tsn, ts_s1, ts_p);
  float tsn_const = compress_hyperbolic_power(tsn, s_Lp100, ts_p);
  tsn = compress_hyperbolic_power(tsn, ts_s, ts_p);
  float crv_tsn_const = 0.0f;
  if (crv_enable == 1) { crv_tsn_const = compress_hyperbolic_power(crv_tsn, s_Lp100, ts_p); crv_tsn = compress_hyperbolic_power(crv_tsn, ts_s, ts_p); }

  if (hc_enable) {
    float hc_ts = pow(1.0f - tsn_const, 1.0f / hc_r_rng);
    float hc_c = (1.0f - tsn_const) * (1.0f - ach_d) + ach_d * tsn_const;
    hc_c *= ach_d * ha_rgb.x;
    float hc_f = hc_r * (hc_c - 2.0f * hc_c * hc_ts) + 1.0f;
    rgb = make_float3(rgb.x, rgb.y * hc_f, rgb.z * hc_f);
  }

  if (hs_rgb_enable) {
    float3 hs_rgb = make_float3(ha_rgb_hs.x * ach_d * pow(tsn_pt, 1.0f / hs_r_rng), ha_rgb_hs.y * ach_d * pow(tsn_pt, 1.0f / hs_g_rng), ha_rgb_hs.z * ach_d * pow(tsn_pt, 1.0f / hs_b_rng));
    float3 hsf = make_float3(hs_rgb.x * hs_r, hs_rgb.y * -hs_g, hs_rgb.z * -hs_b);
    hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
    rgb += hsf;
  }
  if (hs_cmy_enable) {
    float tsn_pt_compl = 1.0f - tsn_pt;
    float3 hs_cmy = make_float3(ha_cmy.x * ach_d * pow(tsn_pt_compl, 1.0f / hs_c_rng), ha_cmy.y * ach_d * pow(tsn_pt_compl, 1.0f / hs_m_rng), ha_cmy.z * ach_d * pow(tsn_pt_compl, 1.0f / hs_y_rng));
    float3 hsf = make_float3(hs_cmy.x * -hs_c, hs_cmy.y * hs_m, hs_cmy.z * hs_y);
    hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
    rgb += hsf;
  }

  float pt_lml_p = 1.0f + 4.0f * (1.0f - tsn_pt) * (pt_lml + pt_lml_r * ha_rgb_hs.x + pt_lml_g * ha_rgb_hs.y + pt_lml_b * ha_rgb_hs.z);
  float ptf = 1.0f;
  if (pt_enable) {
    ptf = 1.0f - pow(tsn_pt, pt_lml_p);
    float pt_lmh_p = (1.0f - ach_d * (pt_lmh_r * ha_rgb_hs.x + pt_lmh_b * ha_rgb_hs.z)) * (1.0f - pt_lmh * ach_d);
    ptf = pow(ptf, pt_lmh_p);
  }

  if (ptm_enable) {
    float ptm_low_f = (ptm_low_st == 0.0f || ptm_low_rng == 0.0f) ? 1.0f : 1.0f + ptm_low * exp(-2.0f * ach_d * ach_d / ptm_low_st) * pow(1.0f - tsn_const, 1.0f / ptm_low_rng);
    float ptm_high_f = (ptm_high_st == 0.0f || ptm_high_rng == 0.0f) ? 1.0f : 1.0f + ptm_high * exp(-2.0f * ach_d * ach_d / ptm_high_st) * pow(tsn_pt, 1.0f / (4.0f * ptm_high_rng));
    ptf *= ptm_low_f * ptm_high_f;
  }

  rgb = rgb * ptf + 1.0f - ptf;
  sat_L = rgb.x * rs_w.x + rgb.y * rs_w.y + rgb.z * rs_w.z;
  float inv_rs_denom = rs_sa - 1.0f;
  if (fabs(inv_rs_denom) < 1e-6f) inv_rs_denom = inv_rs_denom < 0.0f ? -1e-6f : 1e-6f;
  rgb = (sat_L * rs_sa - rgb) / inv_rs_denom;
  rgb = display_gamut_whitepoint(rgb, tsn_const, cwp_lm, display_gamut, cwp);

  if (brlp_enable) {
    float2 brlp_opp = opponent(rgb);
    float brlp_ach_d = hypotf2(brlp_opp) / 4.0f;
    brlp_ach_d = 1.1f * (brlp_ach_d * brlp_ach_d / (brlp_ach_d + 0.1f));
    float3 brlp_ha_rgb = ach_d * ha_rgb;
    float brlp_m = brlp + brlp_r * brlp_ha_rgb.x + brlp_g * brlp_ha_rgb.y + brlp_b * brlp_ha_rgb.z;
    rgb *= pow(2.0f, brlp_m * brlp_ach_d * tsn);
  }

  if (ptl_enable) rgb = make_float3(softplus(rgb.x, ptl_c), softplus(rgb.y, ptl_m), softplus(rgb.z, ptl_y));

  tsn = compress_toe_quadratic(tsn * ts_m2, tn_toe, 0) * ts_dsc;
  if (crv_enable == 1) {
    crv_tsn = compress_toe_quadratic(crv_tsn * ts_m2, tn_toe, 0) * ts_dsc;
    if (eotf == 4) crv_tsn *= 10.0f;
  }
  float3 crv_rgb = make_float3(crv_tsn, crv_tsn, crv_tsn);
  if (crv_enable == 1) crv_rgb = display_gamut_whitepoint(crv_rgb, crv_tsn_const, cwp_lm, display_gamut, cwp);
  rgb *= tsn;

  if (display_gamut == 2) { rgb = vdot(matrix_p3_to_rec2020, clampminf3(rgb, 0.0f)); }
  if (clamp) rgb = clampf3(rgb, 0.0f, 1.0f);

  float eotf_p = 2.0f + eotf * 0.2f;
  if ((eotf > 0) && (eotf < 4)) rgb = spowf3(rgb, 1.0f / eotf_p);
  else if (eotf == 4) rgb = eotf_pq(rgb, 1);
  else if (eotf == 5) rgb = eotf_hlg(rgb, 1);

  if (crv_enable == 1) {
    if ((eotf > 0) && (eotf < 4)) crv_rgb = spowf3(crv_rgb, 1.0f / eotf_p);
    else if (eotf == 4) crv_rgb = eotf_pq(crv_rgb, 1);
    else if (eotf == 5) crv_rgb = eotf_hlg(crv_rgb, 1);
    float3 crv_rgb_dst = make_float3(pos.y - crv_rgb.x * res.y, pos.y - crv_rgb.y * res.y, pos.y - crv_rgb.z * res.y);
    float crv_w0 = 0.35f;
    crv_rgb_dst.x = exp(-crv_rgb_dst.x * crv_rgb_dst.x * crv_w0);
    crv_rgb_dst.y = exp(-crv_rgb_dst.y * crv_rgb_dst.y * crv_w0);
    crv_rgb_dst.z = exp(-crv_rgb_dst.z * crv_rgb_dst.z * crv_w0);
    crv_rgb_dst = clampf3(crv_rgb_dst, 0.0f, 1.0f);
    rgb = rgb * (1.0f - crv_rgb_dst) + make_float3(1.0f, 1.0f, 1.0f) * crv_rgb_dst;
  }
  return rgb;
}


kernel void OpenDRTKernel(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant OpenDRTParams& p [[buffer(2)]],
                          constant OpenDRTDerivedParams& d [[buffer(5)]],
                          constant int& width [[buffer(3)]],
                          constant int& height [[buffer(4)]],
                          constant int& srcRowFloats [[buffer(6)]],
                          constant int& dstRowFloats [[buffer(7)]],
                          uint2 gid [[thread_position_in_grid]]) {
  // Flat interleaved RGBA layout: one thread computes one destination pixel.
  const int x = static_cast<int>(gid.x);
  const int y = static_cast<int>(gid.y);
  if (x >= width || y >= height) return;
  const int iSrc = y * srcRowFloats + x * 4;
  const int iDst = y * dstRowFloats + x * 4;
  float3 rgb = make_float3(src[iSrc + 0], src[iSrc + 1], src[iSrc + 2]);
  rgb = openDRTTransform(width, height, x, y, rgb, p, d);
  dst[iDst + 0] = rgb.x;
  dst[iDst + 1] = rgb.y;
  dst[iDst + 2] = rgb.z;
  dst[iDst + 3] = src[iSrc + 3];
}


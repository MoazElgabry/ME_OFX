#pragma once

struct OpenDRTParams {
  int in_gamut;
  int in_oetf;

  float tn_Lp;
  float tn_gb;
  float pt_hdr;
  float tn_Lg;
  int crv_enable;
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

  int cwp;
  int display_gamut;
  int eotf;
  int tn_su;
  int clamp;
  float cwp_lm;
};

struct OpenDRTDerivedParams {
  int enabled;
  float ts_x1;
  float ts_y1;
  float ts_x0;
  float ts_y0;
  float ts_s0;
  float ts_p;
  float ts_s10;
  float ts_m1;
  float ts_m2;
  float ts_s;
  float ts_dsc;
  float pt_cmp_Lf;
  float s_Lp100;
  float ts_s1;
};

n 2048, peak bandwidth 256 GB/s

version         GFLOP/s   traffic GB   required GB/s   % of peak   verdict
v1_naive            93.5        68.72           373.8      146.0%   cache is absorbing most of the traffic
v2_coalesced       771.6        68.72          3086.5     1205.5%   cache is absorbing most of the traffic
v3_shared          979.3         2.15           122.4       47.8%   partly bandwidth limited
v4_1d_tiling      2503.4         1.07           156.5       61.1%   partly bandwidth limited
v5_2d_tiling      3633.1         0.54           113.5       44.3%   partly bandwidth limited
v6_vectorized     6098.0         0.54           190.6       74.4%   DRAM bandwidth limited

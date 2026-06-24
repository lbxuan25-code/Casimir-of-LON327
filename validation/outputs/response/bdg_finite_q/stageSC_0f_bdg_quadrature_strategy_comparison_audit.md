# stageSC_0f_bdg_quadrature_strategy_comparison_audit

- status: FAILED
- formal Casimir ran: False
- diagnostic only: True
- Schur assembly: one Schur inversion after composite linear-kernel accumulation
- interpretation: No tested arbitrary-q strategy simultaneously reaches onsite_s contact, bare-Ward, and AP-Ward 1e-6 limits.
- best observed (not necessarily passing): multi_origin_dense N=72

## Strategy comparison summary

| pairing | strategy | N | q | contact closure max | bare Ward max | AP Ward max | status |
| ------- | -------- | -: | - | ------------------: | ------------: | ----------: | ------ |
| onsite_s | ordinary_uniform | 24 | (0.01,0) | 0.001288786 | 0.019331079 | 0.0012383167 | FAILED |
| onsite_s | ordinary_uniform | 24 | (0.01,0.01) | 0.0012844737 | 0.019316796 | 0.0011924855 | FAILED |
| onsite_s | ordinary_uniform | 36 | (0.01,0) | 0.0016731326 | 0.024156198 | 0.0015116516 | FAILED |
| onsite_s | ordinary_uniform | 36 | (0.01,0.01) | 0.0016692499 | 0.024084768 | 0.0014015103 | FAILED |
| onsite_s | ordinary_uniform | 48 | (0.01,0) | 0.00015544559 | 0.021379332 | 0.00025461269 | FAILED |
| onsite_s | ordinary_uniform | 48 | (0.01,0.01) | 0.00015230487 | 0.021353927 | 0.00044609778 | FAILED |
| onsite_s | multi_origin_symmetric | 24 | (0.01,0) | 0.0012809503 | 0.019339619 | 0.0012316831 | FAILED |
| onsite_s | multi_origin_symmetric | 24 | (0.01,0.01) | 0.0012721614 | 0.019333403 | 0.0011834849 | FAILED |
| onsite_s | multi_origin_symmetric | 36 | (0.01,0) | 0.0016550454 | 0.024141295 | 0.0014946446 | FAILED |
| onsite_s | multi_origin_symmetric | 36 | (0.01,0.01) | 0.001646576 | 0.024055016 | 0.0013812123 | FAILED |
| onsite_s | multi_origin_symmetric | 48 | (0.01,0) | 0.00014201545 | 0.021390674 | 0.00023227185 | FAILED |
| onsite_s | multi_origin_symmetric | 48 | (0.01,0.01) | 0.00013596722 | 0.021376055 | 0.00039741832 | FAILED |
| onsite_s | grid_step_commensurate_reference | 24 | (0.5235988,0) | 1.6654438e-16 | 0.018075842 | 3.4483054e-16 | FAILED |
| onsite_s | grid_step_commensurate_reference | 24 | (0.5235988,0.5235988) | 3.8857911e-16 | 0.014245032 | 1.7347238e-16 | FAILED |
| onsite_s | grid_step_commensurate_reference | 36 | (0.3490659,0) | 1.1102313e-16 | 0.022027144 | 1.2056624e-16 | FAILED |
| onsite_s | grid_step_commensurate_reference | 36 | (0.3490659,0.3490659) | 1.6653346e-16 | 0.017997121 | 2.2898357e-16 | FAILED |
| onsite_s | grid_step_commensurate_reference | 48 | (0.2617994,0) | 2.7756721e-17 | 0.021234817 | 1.3292537e-16 | FAILED |
| onsite_s | grid_step_commensurate_reference | 48 | (0.2617994,0.2617994) | 8.3266942e-17 | 0.017649753 | 7.3420104e-17 | FAILED |
| onsite_s | high_resolution_uniform | 48 | (0.01,0) | 0.00015544559 | 0.021379332 | 0.00025461269 | FAILED |
| onsite_s | high_resolution_uniform | 48 | (0.01,0.01) | 0.00015230487 | 0.021353927 | 0.00044609778 | FAILED |
| onsite_s | high_resolution_uniform | 72 | (0.01,0) | 0.00016974605 | 0.021760097 | 0.00014623913 | FAILED |
| onsite_s | high_resolution_uniform | 72 | (0.01,0.01) | 0.00016872719 | 0.021718081 | 0.00023192909 | FAILED |
| onsite_s | high_resolution_uniform | 96 | (0.01,0) | 0.00014173699 | 0.021284693 | 0.00019774995 | FAILED |
| onsite_s | high_resolution_uniform | 96 | (0.01,0.01) | 0.00014141182 | 0.021256247 | 0.00034859088 | FAILED |
| onsite_s | multi_origin_dense | 36 | (0.01,0) | 0.0016550454 | 0.024141295 | 0.0014946446 | FAILED |
| onsite_s | multi_origin_dense | 36 | (0.01,0.01) | 0.001646576 | 0.024055016 | 0.0013812123 | FAILED |
| onsite_s | multi_origin_dense | 48 | (0.01,0) | 0.00014201545 | 0.021390674 | 0.00023227185 | FAILED |
| onsite_s | multi_origin_dense | 48 | (0.01,0.01) | 0.00013596722 | 0.021376055 | 0.00039741832 | FAILED |
| onsite_s | multi_origin_dense | 72 | (0.01,0) | 0.00016519423 | 0.02175689 | 0.00014185955 | FAILED |
| onsite_s | multi_origin_dense | 72 | (0.01,0.01) | 0.00016312967 | 0.021711813 | 0.00022799655 | FAILED |
| spm | ordinary_uniform | 24 | (0.01,0) | 0.0028826581 | 0.016560051 | 0.0028673375 | FAILED |
| spm | ordinary_uniform | 24 | (0.01,0.01) | 0.0028770595 | 0.016558438 | 0.0028469245 | FAILED |
| spm | ordinary_uniform | 36 | (0.01,0) | 0.0017357667 | 0.023191945 | 0.0018053821 | FAILED |
| spm | ordinary_uniform | 36 | (0.01,0.01) | 0.0021134096 | 0.022917782 | 0.0022402973 | FAILED |
| spm | ordinary_uniform | 48 | (0.01,0) | 0.0020752991 | 0.017899091 | 0.0020100425 | FAILED |
| spm | ordinary_uniform | 48 | (0.01,0.01) | 0.0020559103 | 0.017903573 | 0.0019300292 | FAILED |
| spm | multi_origin_symmetric | 24 | (0.01,0) | 0.0028415389 | 0.01659519 | 0.0028258194 | FAILED |
| spm | multi_origin_symmetric | 24 | (0.01,0.01) | 0.0028213557 | 0.016631288 | 0.0027896309 | FAILED |
| spm | multi_origin_symmetric | 36 | (0.01,0) | 0.0022002289 | 0.023308412 | 0.0022414681 | FAILED |
| spm | multi_origin_symmetric | 36 | (0.01,0.01) | 0.0029166377 | 0.023077248 | 0.0029652624 | FAILED |
| spm | multi_origin_symmetric | 48 | (0.01,0) | 0.0019586759 | 0.017987525 | 0.0018935327 | FAILED |
| spm | multi_origin_symmetric | 48 | (0.01,0.01) | 0.0018968096 | 0.018075749 | 0.001773014 | FAILED |
| spm | grid_step_commensurate_reference | 24 | (0.5235988,0) | 1.5367793e-18 | 0.012384532 | 4.8019605e-16 | FAILED |
| spm | grid_step_commensurate_reference | 24 | (0.5235988,0.5235988) | 2.2205108e-16 | 0.010161768 | 1.5352372e-16 | FAILED |
| spm | grid_step_commensurate_reference | 36 | (0.3490659,0) | 2.2204475e-16 | 0.016432457 | 6.9612637e-17 | FAILED |
| spm | grid_step_commensurate_reference | 36 | (0.3490659,0.3490659) | 2.4980018e-16 | 0.012305762 | 8.8472066e-17 | FAILED |
| spm | grid_step_commensurate_reference | 48 | (0.2617994,0) | 2.7756648e-17 | 0.015354519 | 5.5803513e-17 | FAILED |
| spm | grid_step_commensurate_reference | 48 | (0.2617994,0.2617994) | 2.7756613e-17 | 0.013417687 | 7.7196076e-17 | FAILED |
| spm | high_resolution_uniform | 48 | (0.01,0) | 0.0020752991 | 0.017899091 | 0.0020100425 | FAILED |
| spm | high_resolution_uniform | 48 | (0.01,0.01) | 0.0020559103 | 0.017903573 | 0.0019300292 | FAILED |
| spm | high_resolution_uniform | 72 | (0.01,0) | 0.00016760952 | 0.019967224 | 0.00031655806 | FAILED |
| spm | high_resolution_uniform | 72 | (0.01,0.01) | 4.59606e-05 | 0.019807269 | 0.00055238135 | FAILED |
| spm | high_resolution_uniform | 96 | (0.01,0) | 0.00059175367 | 0.018436444 | 0.00055556534 | FAILED |
| spm | high_resolution_uniform | 96 | (0.01,0.01) | 0.00073994841 | 0.018296718 | 0.0007083739 | FAILED |
| spm | multi_origin_dense | 36 | (0.01,0) | 0.0022002289 | 0.023308412 | 0.0022414681 | FAILED |
| spm | multi_origin_dense | 36 | (0.01,0.01) | 0.0029166377 | 0.023077248 | 0.0029652624 | FAILED |
| spm | multi_origin_dense | 48 | (0.01,0) | 0.0019586759 | 0.017987525 | 0.0018935327 | FAILED |
| spm | multi_origin_dense | 48 | (0.01,0.01) | 0.0018968096 | 0.018075749 | 0.001773014 | FAILED |
| spm | multi_origin_dense | 72 | (0.01,0) | 0.00023223879 | 0.020043219 | 0.00026074054 | FAILED |
| spm | multi_origin_dense | 72 | (0.01,0.01) | 0.00043878848 | 0.019906301 | 0.00047636058 | FAILED |
| dwave | ordinary_uniform | 24 | (0.01,0) | 0.0017993343 | 0.012149815 | 0.0017800342 | FAILED |
| dwave | ordinary_uniform | 24 | (0.01,0.01) | 0.0017955315 | 0.012126941 | 0.0017606776 | FAILED |
| dwave | ordinary_uniform | 36 | (0.01,0) | 0.0016664479 | 0.018433326 | 0.0016534787 | FAILED |
| dwave | ordinary_uniform | 36 | (0.01,0.01) | 0.0016347014 | 0.018230441 | 0.0016110588 | FAILED |
| dwave | ordinary_uniform | 48 | (0.01,0) | 0.0018063658 | 0.012596246 | 0.0017346724 | FAILED |
| dwave | ordinary_uniform | 48 | (0.01,0.01) | 0.0017901813 | 0.01258232 | 0.0016578285 | FAILED |
| dwave | multi_origin_symmetric | 24 | (0.01,0) | 0.0017847809 | 0.012172106 | 0.0017676817 | FAILED |
| dwave | multi_origin_symmetric | 24 | (0.01,0.01) | 0.0017763019 | 0.01216999 | 0.0017460856 | FAILED |
| dwave | multi_origin_symmetric | 36 | (0.01,0) | 0.0017955424 | 0.018478141 | 0.0017716024 | FAILED |
| dwave | multi_origin_symmetric | 36 | (0.01,0.01) | 0.0017591493 | 0.018333583 | 0.0017203314 | FAILED |
| dwave | multi_origin_symmetric | 48 | (0.01,0) | 0.0017096599 | 0.012666601 | 0.0016346449 | FAILED |
| dwave | multi_origin_symmetric | 48 | (0.01,0.01) | 0.0016646954 | 0.01272284 | 0.00152823 | FAILED |
| dwave | grid_step_commensurate_reference | 24 | (0.5235988,0) | 1.1102977e-16 | 0.0063496717 | 0.00087408445 | FAILED |
| dwave | grid_step_commensurate_reference | 24 | (0.5235988,0.5235988) | 1.6653979e-16 | 0.0045138882 | 0.00083398815 | FAILED |
| dwave | grid_step_commensurate_reference | 36 | (0.3490659,0) | 1.6653367e-16 | 0.0086264138 | 0.00063619755 | FAILED |
| dwave | grid_step_commensurate_reference | 36 | (0.3490659,0.3490659) | 2.2204461e-16 | 0.0063681954 | 0.00063095811 | FAILED |
| dwave | grid_step_commensurate_reference | 48 | (0.2617994,0) | 2.7757233e-17 | 0.0083779385 | 0.0004723741 | FAILED |
| dwave | grid_step_commensurate_reference | 48 | (0.2617994,0.2617994) | 8.326675e-17 | 0.006588706 | 0.00047359299 | FAILED |
| dwave | high_resolution_uniform | 48 | (0.01,0) | 0.0018063658 | 0.012596246 | 0.0017346724 | FAILED |
| dwave | high_resolution_uniform | 48 | (0.01,0.01) | 0.0017901813 | 0.01258232 | 0.0016578285 | FAILED |
| dwave | high_resolution_uniform | 72 | (0.01,0) | 0.00040670721 | 0.014889474 | 0.00042164081 | FAILED |
| dwave | high_resolution_uniform | 72 | (0.01,0.01) | 4.3352628e-05 | 0.014774305 | 0.00023579578 | FAILED |
| dwave | high_resolution_uniform | 96 | (0.01,0) | 0.0013574536 | 0.013565732 | 0.0013414342 | FAILED |
| dwave | high_resolution_uniform | 96 | (0.01,0.01) | 0.0013609298 | 0.013506201 | 0.0013309675 | FAILED |
| dwave | multi_origin_dense | 36 | (0.01,0) | 0.0017955424 | 0.018478141 | 0.0017716024 | FAILED |
| dwave | multi_origin_dense | 36 | (0.01,0.01) | 0.0017591493 | 0.018333583 | 0.0017203314 | FAILED |
| dwave | multi_origin_dense | 48 | (0.01,0) | 0.0017096599 | 0.012666601 | 0.0016346449 | FAILED |
| dwave | multi_origin_dense | 48 | (0.01,0.01) | 0.0016646954 | 0.01272284 | 0.00152823 | FAILED |
| dwave | multi_origin_dense | 72 | (0.01,0) | 7.9915261e-05 | 0.01491351 | 0.00015505359 | FAILED |
| dwave | multi_origin_dense | 72 | (0.01,0.01) | 6.8169218e-05 | 0.014833915 | 7.7105121e-05 | FAILED |

## onsite_s fixed-q strategy ranking

| rank | strategy | N | origins | contact closure | bare Ward | AP Ward | cost |
| ---: | -------- | -: | ------: | --------------: | --------: | ------: | ---: |
| 1 | multi_origin_dense | 72 | 7 | 0.00016519423 | 0.02175689 | 0.00014185955 | 36288 |
| 2 | high_resolution_uniform | 72 | 1 | 0.00016974605 | 0.021760097 | 0.00014623913 | 5184 |
| 3 | high_resolution_uniform | 96 | 1 | 0.00014173699 | 0.021284693 | 0.00019774995 | 9216 |
| 4 | multi_origin_dense | 72 | 7 | 0.00016312967 | 0.021711813 | 0.00022799655 | 36288 |
| 5 | high_resolution_uniform | 72 | 1 | 0.00016872719 | 0.021718081 | 0.00023192909 | 5184 |
| 6 | multi_origin_symmetric | 48 | 7 | 0.00014201545 | 0.021390674 | 0.00023227185 | 16128 |
| 7 | multi_origin_dense | 48 | 7 | 0.00014201545 | 0.021390674 | 0.00023227185 | 16128 |
| 8 | ordinary_uniform | 48 | 1 | 0.00015544559 | 0.021379332 | 0.00025461269 | 2304 |
| 9 | high_resolution_uniform | 48 | 1 | 0.00015544559 | 0.021379332 | 0.00025461269 | 2304 |
| 10 | high_resolution_uniform | 96 | 1 | 0.00014141182 | 0.021256247 | 0.00034859088 | 9216 |
| 11 | multi_origin_symmetric | 48 | 7 | 0.00013596722 | 0.021376055 | 0.00039741832 | 16128 |
| 12 | multi_origin_dense | 48 | 7 | 0.00013596722 | 0.021376055 | 0.00039741832 | 16128 |
| 13 | ordinary_uniform | 48 | 1 | 0.00015230487 | 0.021353927 | 0.00044609778 | 2304 |
| 14 | high_resolution_uniform | 48 | 1 | 0.00015230487 | 0.021353927 | 0.00044609778 | 2304 |
| 15 | multi_origin_symmetric | 24 | 7 | 0.0012721614 | 0.019333403 | 0.0011834849 | 4032 |
| 16 | ordinary_uniform | 24 | 1 | 0.0012844737 | 0.019316796 | 0.0011924855 | 576 |
| 17 | multi_origin_symmetric | 24 | 7 | 0.0012809503 | 0.019339619 | 0.0012316831 | 4032 |
| 18 | ordinary_uniform | 24 | 1 | 0.001288786 | 0.019331079 | 0.0012383167 | 576 |
| 19 | multi_origin_symmetric | 36 | 7 | 0.001646576 | 0.024055016 | 0.0013812123 | 9072 |
| 20 | multi_origin_dense | 36 | 7 | 0.001646576 | 0.024055016 | 0.0013812123 | 9072 |
| 21 | ordinary_uniform | 36 | 1 | 0.0016692499 | 0.024084768 | 0.0014015103 | 1296 |
| 22 | multi_origin_symmetric | 36 | 7 | 0.0016550454 | 0.024141295 | 0.0014946446 | 9072 |
| 23 | multi_origin_dense | 36 | 7 | 0.0016550454 | 0.024141295 | 0.0014946446 | 9072 |
| 24 | ordinary_uniform | 36 | 1 | 0.0016731326 | 0.024156198 | 0.0015116516 | 1296 |

## spm/dwave monitor

| pairing | strategy | N | q | AP Ward | sigma diag min | offdiag rel | max sigma tilde |
| ------- | -------- | -: | - | ------: | -------------: | ----------: | --------------: |
| spm | ordinary_uniform | 24 | (0.01,0) | 0.0028673375 | 36.526236 | 1.9260563e-17 | 37.068993 |
| spm | ordinary_uniform | 24 | (0.01,0.01) | 0.0028469245 | 36.519927 | 0.010042559 | 36.519927 |
| spm | ordinary_uniform | 36 | (0.01,0) | 0.0018053821 | 24.360799 | 1.4749351e-16 | 36.02289 |
| spm | ordinary_uniform | 36 | (0.01,0.01) | 0.0022402973 | 23.862536 | 0.33224341 | 23.862536 |
| spm | ordinary_uniform | 48 | (0.01,0) | 0.0020100425 | 35.685805 | 9.7732204e-17 | 37.512623 |
| spm | ordinary_uniform | 48 | (0.01,0.01) | 0.0019300292 | 35.682383 | 0.037982835 | 35.682383 |
| spm | multi_origin_symmetric | 24 | (0.01,0) | 0.0028258194 | 36.453893 | 6.6349296e-18 | 37.066708 |
| spm | multi_origin_symmetric | 24 | (0.01,0.01) | 0.0027896309 | 36.419326 | 0.011631146 | 36.419326 |
| spm | multi_origin_symmetric | 36 | (0.01,0) | 0.0022414681 | 22.482155 | 2.640374e-16 | 35.163636 |
| spm | multi_origin_symmetric | 36 | (0.01,0.01) | 0.0029652624 | 21.056448 | 0.49407775 | 21.056448 |
| spm | multi_origin_symmetric | 48 | (0.01,0) | 0.0018935327 | 35.406464 | 4.3691764e-17 | 37.503 |
| spm | multi_origin_symmetric | 48 | (0.01,0.01) | 0.001773014 | 35.295258 | 0.045684385 | 35.295258 |
| spm | grid_step_commensurate_reference | 24 | (0.5235988,0) | 4.8019605e-16 | 0.07107793 | 1.1510817e-17 | 23.682448 |
| spm | grid_step_commensurate_reference | 24 | (0.5235988,0.5235988) | 1.5352372e-16 | 0.96741327 | 0.95905535 | 0.96741327 |
| spm | grid_step_commensurate_reference | 36 | (0.3490659,0) | 6.9612637e-17 | 0.18491995 | 9.5299761e-16 | 12.031103 |
| spm | grid_step_commensurate_reference | 36 | (0.3490659,0.3490659) | 8.8472066e-17 | 3.1963303 | 0.97287963 | 3.1963303 |
| spm | grid_step_commensurate_reference | 48 | (0.2617994,0) | 5.5803513e-17 | 0.28131248 | 5.6243737e-17 | 24.284943 |
| spm | grid_step_commensurate_reference | 48 | (0.2617994,0.2617994) | 7.7196076e-17 | 0.77604395 | 0.80192852 | 0.77604395 |
| spm | high_resolution_uniform | 48 | (0.01,0) | 0.0020100425 | 35.685805 | 9.7732204e-17 | 37.512623 |
| spm | high_resolution_uniform | 48 | (0.01,0.01) | 0.0019300292 | 35.682383 | 0.037982835 | 35.682383 |
| spm | high_resolution_uniform | 72 | (0.01,0) | 0.00031655806 | 28.300716 | 2.3697554e-16 | 34.083627 |
| spm | high_resolution_uniform | 72 | (0.01,0.01) | 0.00055238135 | 29.328891 | 0.17017393 | 29.328891 |
| spm | high_resolution_uniform | 96 | (0.01,0) | 0.00055556534 | 29.654592 | 2.2453192e-16 | 33.86485 |
| spm | high_resolution_uniform | 96 | (0.01,0.01) | 0.0007083739 | 30.058962 | 0.12008799 | 30.058962 |
| spm | multi_origin_dense | 36 | (0.01,0) | 0.0022414681 | 22.482155 | 2.640374e-16 | 35.163636 |
| spm | multi_origin_dense | 36 | (0.01,0.01) | 0.0029652624 | 21.056448 | 0.49407775 | 21.056448 |
| spm | multi_origin_dense | 48 | (0.01,0) | 0.0018935327 | 35.406464 | 4.3691764e-17 | 37.503 |
| spm | multi_origin_dense | 48 | (0.01,0.01) | 0.001773014 | 35.295258 | 0.045684385 | 35.295258 |
| spm | multi_origin_dense | 72 | (0.01,0) | 0.00026074054 | 28.084997 | 1.6927176e-16 | 34.455411 |
| spm | multi_origin_dense | 72 | (0.01,0.01) | 0.00047636058 | 27.923272 | 0.21939246 | 27.923272 |
| dwave | ordinary_uniform | 24 | (0.01,0) | 0.0017800342 | 34.581782 | 8.756342e-16 | 36.802446 |
| dwave | ordinary_uniform | 24 | (0.01,0.01) | 0.0017606776 | 34.735125 | 0.055623297 | 34.735125 |
| dwave | ordinary_uniform | 36 | (0.01,0) | 0.0016534787 | 24.12035 | 4.8190702e-16 | 35.352317 |
| dwave | ordinary_uniform | 36 | (0.01,0.01) | 0.0016110588 | 25.643934 | 0.32912865 | 25.643934 |
| dwave | ordinary_uniform | 48 | (0.01,0) | 0.0017346724 | 34.812877 | 2.6230032e-16 | 37.328836 |
| dwave | ordinary_uniform | 48 | (0.01,0.01) | 0.0016578285 | 34.914567 | 0.061531684 | 34.914567 |
| dwave | multi_origin_symmetric | 24 | (0.01,0) | 0.0017676817 | 34.542128 | 5.3585009e-16 | 36.798614 |
| dwave | multi_origin_symmetric | 24 | (0.01,0.01) | 0.0017460856 | 34.682368 | 0.05690298 | 34.682368 |
| dwave | multi_origin_symmetric | 36 | (0.01,0) | 0.0017716024 | 23.567107 | 3.032624e-16 | 35.365243 |
| dwave | multi_origin_symmetric | 36 | (0.01,0.01) | 0.0017203314 | 25.161539 | 0.33806013 | 25.161539 |
| dwave | multi_origin_symmetric | 48 | (0.01,0) | 0.0016346449 | 34.517753 | 1.503563e-16 | 37.322482 |
| dwave | multi_origin_symmetric | 48 | (0.01,0.01) | 0.00152823 | 34.569299 | 0.06814547 | 34.569299 |
| dwave | grid_step_commensurate_reference | 24 | (0.5235988,0) | 0.00087408445 | 0.23678747 | 1.8759298e-16 | 22.003716 |
| dwave | grid_step_commensurate_reference | 24 | (0.5235988,0.5235988) | 0.00083398815 | 0.06066378 | 2.2157335 | 0.13441477 |
| dwave | grid_step_commensurate_reference | 36 | (0.3490659,0) | 0.00063619755 | 0.35434635 | 2.0992168e-16 | 16.947537 |
| dwave | grid_step_commensurate_reference | 36 | (0.3490659,0.3490659) | 0.00063095811 | 2.157557 | 0.87858386 | 2.157557 |
| dwave | grid_step_commensurate_reference | 48 | (0.2617994,0) | 0.0004723741 | 0.44652577 | 3.2679713e-16 | 26.07016 |
| dwave | grid_step_commensurate_reference | 48 | (0.2617994,0.2617994) | 0.00047359299 | 1.841434 | 0.82467748 | 1.841434 |
| dwave | high_resolution_uniform | 48 | (0.01,0) | 0.0017346724 | 34.812877 | 2.6230032e-16 | 37.328836 |
| dwave | high_resolution_uniform | 48 | (0.01,0.01) | 0.0016578285 | 34.914567 | 0.061531684 | 34.914567 |
| dwave | high_resolution_uniform | 72 | (0.01,0) | 0.00042164081 | 30.485562 | 2.2365544e-16 | 36.080441 |
| dwave | high_resolution_uniform | 72 | (0.01,0.01) | 0.00023579578 | 29.418951 | 0.20795326 | 29.418951 |
| dwave | high_resolution_uniform | 96 | (0.01,0) | 0.0013414342 | 33.524897 | 1.300375e-16 | 36.851998 |
| dwave | high_resolution_uniform | 96 | (0.01,0.01) | 0.0013309675 | 33.801954 | 0.083281212 | 33.801954 |
| dwave | multi_origin_dense | 36 | (0.01,0) | 0.0017716024 | 23.567107 | 3.032624e-16 | 35.365243 |
| dwave | multi_origin_dense | 36 | (0.01,0.01) | 0.0017203314 | 25.161539 | 0.33806013 | 25.161539 |
| dwave | multi_origin_dense | 48 | (0.01,0) | 0.0016346449 | 34.517753 | 1.503563e-16 | 37.322482 |
| dwave | multi_origin_dense | 48 | (0.01,0.01) | 0.00152823 | 34.569299 | 0.06814547 | 34.569299 |
| dwave | multi_origin_dense | 72 | (0.01,0) | 0.00015505359 | 29.152568 | 2.1170338e-16 | 35.03477 |
| dwave | multi_origin_dense | 72 | (0.01,0.01) | 7.7105121e-05 | 29.739894 | 0.18558772 | 29.739894 |

## Recommended strategy

| recommended strategy | N | origins | reason |
| -------------------- | -: | ------: | ------ |
| none | - | - | No tested arbitrary-q strategy simultaneously reaches onsite_s contact, bare-Ward, and AP-Ward 1e-6 limits. |

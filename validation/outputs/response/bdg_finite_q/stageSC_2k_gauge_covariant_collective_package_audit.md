# stageSC_2k_gauge_covariant_collective_package_audit

- status: FAILED_STAGE2K_CONTROL_REGRESSION
- diagnostic only: True
- formal Casimir ran: False
- production default modified: False
- valid for Casimir input: False

This stage tests a forward-derived gauge-covariant collective package. LSQ is used only as a diagnostic reference and is not used to define the package.

## Case summary

| pairing | N | q | baseline | LSQ | StageSC-2j | best package | best Ward | best C | lambda | alpha |
| ------- | -: | - | -------: | --: | ---------: | ------------ | --------: | ------ | ------ | ----- |
| dwave | 24 | (0.5235988,0) | 0.00087408578 | 1.2559842e-17 | 0.00055798548 | pkg_bubble_spatial_with_i | 0.0046665034 | bubble | spatial | with_i |
| dwave | 24 | (0.5235988,0.5235988) | 0.00083397642 | 1.4696624e-16 | 0.00025459645 | pkg_bubble_spatial_with_i | 0.0024337104 | bubble | spatial | with_i |
| dwave | 36 | (0.3490659,0) | 0.00063617515 | 6.0921378e-18 | 0.00057359818 | pkg_bubble_spatial_with_i | 0.0067920445 | bubble | spatial | with_i |
| dwave | 36 | (0.3490659,0.3490659) | 0.00063094848 | 1.8229414e-17 | 0.00026643857 | pkg_bubble_spatial_with_i | 0.0038960603 | bubble | spatial | with_i |
| onsite_s | 24 | (0.5235988,0) | 3.4483054e-16 | 3.469447e-18 | 3.4543048e-16 | pkg_bond_metric_spatial_with_i | 0.0097004212 | bond_metric | spatial | with_i |
| spm | 24 | (0.5235988,0) | 4.8019605e-16 | 1.0062046e-18 | 4.7569626e-16 | pkg_bubble_spatial_with_i | 0.0063335212 | bubble | spatial | with_i |
| dwave_const_form | 24 | (0.5235988,0) | 2.8694333e-16 | 3.4179953e-17 | 2.8509106e-16 | pkg_bubble_spatial_with_i | 0.005971454 | bubble | spatial | with_i |

## Dwave package detail

| N | q | variant | Ward | vs baseline | vs 2j | cond | dAA | dAeta | detaA | detaeta | C | lambda | alpha |
| -: | - | ------- | ---: | ----------: | ----: | ---: | --: | -----: | ----: | ------: | - | ------ | ----- |
| 24 | (0.5235988,0) | pkg_total_spatial_no_i | 0.012806682 | 0.068252323 | 0.043569872 | 1.0388571 | 0.014234413 | 0.093164051 | 0.093164051 | 0 | collective_total | spatial | no_i |
| 24 | (0.5235988,0) | pkg_total_spatial_with_i | 0.011902732 | 0.073435728 | 0.046878774 | 1.0388571 | 0.014234413 | 0.093164051 | 0.093164051 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0) | pkg_total_spacetime_omega_plus_no_i | 0.012804344 | 0.068264786 | 0.043577827 | 1.0388571 | 0.014229223 | 0.093147065 | 0.093147065 | 0 | collective_total | spacetime | no_i |
| 24 | (0.5235988,0) | pkg_total_spacetime_omega_plus_with_i | 0.011900388 | 0.073450189 | 0.046888006 | 1.0388571 | 0.014229223 | 0.093147065 | 0.093147065 | 0 | collective_total | spacetime | with_i |
| 24 | (0.5235988,0) | pkg_total_spacetime_omega_minus_with_i | 0.011909711 | 0.073392694 | 0.046851303 | 1.0388571 | 0.014249999 | 0.093215042 | 0.093215042 | 0 | collective_total | spacetime | with_i |
| 24 | (0.5235988,0) | pkg_total_spacetime_omega_zero_with_i | 0.011905048 | 0.073421441 | 0.046869654 | 1.0388571 | 0.014239605 | 0.093181041 | 0.093181041 | 0 | collective_total | spacetime | with_i |
| 24 | (0.5235988,0) | pkg_bubble_spatial_with_i | 0.0046665034 | 0.18731065 | 0.1195725 | 1.0126526 | 0.044268269 | 0.28973515 | 0.28973515 | 3.5441169 | bubble | spatial | with_i |
| 24 | (0.5235988,0) | pkg_counterterm_spatial_with_i | 0.0077594081 | 0.11264851 | 0.07191083 | 1 | 0.058502682 | 0.38289916 | 0.38289916 | 2.6650892 | counterterm | spatial | with_i |
| 24 | (0.5235988,0) | pkg_bond_metric_spatial_with_i | 5.4884993 | 0.0001592577 | 0.00010166449 | 13320.939 | 0.056543213 | 0.52783876 | 0.52783876 | 4.350611 | bond_metric | spatial | with_i |
| 24 | (0.5235988,0) | pkg_hybrid_bond_metric_spatial_with_i | 0.0060738757 | 0.14390907 | 0.091866463 | 1.6152178 | 0.012274945 | 0.38477306 | 0.38493611 | 3.4847466 | hybrid | spatial | with_i |
| 24 | (0.5235988,0) | pkg_total_spatial_with_i_no_AA | 0.019355853 | 0.045158732 | 0.028827739 | 1.0388571 | 0 | 0.093164051 | 0.093164051 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0) | pkg_total_spatial_with_i_only_mixed | 0.019355853 | 0.045158732 | 0.028827739 | 1.0388571 | 0 | 0.093164051 | 0.093164051 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0) | pkg_total_spatial_with_i_etaeta_plus_mixed_no_AA | 0.019355853 | 0.045158732 | 0.028827739 | 1.0388571 | 0 | 0.093164051 | 0.093164051 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0) | pkg_total_spatial_with_i_AA_plus_mixed_no_etaeta | 0.011902732 | 0.073435728 | 0.046878774 | 1.0388571 | 0.014234413 | 0.093164051 | 0.093164051 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spatial_no_i | 0.0090789418 | 0.091858329 | 0.028042525 | 1.1028003 | 0.010604769 | 0.098158815 | 0.098158815 | 0 | collective_total | spatial | no_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spatial_with_i | 0.0082065804 | 0.10162289 | 0.031023452 | 1.1028003 | 0.010604769 | 0.098158815 | 0.098158815 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spacetime_omega_plus_no_i | 0.0090781199 | 0.091866645 | 0.028045064 | 1.1028003 | 0.010602835 | 0.098149866 | 0.098149866 | 0 | collective_total | spacetime | no_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spacetime_omega_plus_with_i | 0.008205755 | 0.10163311 | 0.031026572 | 1.1028003 | 0.010602835 | 0.098149866 | 0.098149866 | 0 | collective_total | spacetime | with_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spacetime_omega_minus_with_i | 0.0082090529 | 0.10159228 | 0.031014108 | 1.1028003 | 0.010610573 | 0.098185673 | 0.098185673 | 0 | collective_total | spacetime | with_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spacetime_omega_zero_with_i | 0.0082074036 | 0.1016127 | 0.03102034 | 1.1028003 | 0.010606703 | 0.098167766 | 0.098167766 | 0 | collective_total | spacetime | with_i |
| 24 | (0.5235988,0.5235988) | pkg_bubble_spatial_with_i | 0.0024337104 | 0.34267694 | 0.10461247 | 1.0530155 | 0.018646572 | 0.17259336 | 0.17259336 | 3.5441169 | bubble | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_counterterm_spatial_with_i | 0.0058608412 | 0.14229637 | 0.043440257 | 1 | 0.029251341 | 0.27075059 | 0.27075059 | 2.3199022 | counterterm | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_bond_metric_spatial_with_i | 0.0051452681 | 0.16208609 | 0.049481668 | 1.5166134e+16 | 0.027291873 | 0.36360558 | 0.36360558 | 4.0734647 | bond_metric | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_hybrid_bond_metric_spatial_with_i | 0.0040276766 | 0.20706141 | 0.06321174 | 1.9547179 | 0.0086453002 | 0.27391991 | 0.27306707 | 3.4274677 | hybrid | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spatial_with_i_no_AA | 0.013759224 | 0.060612169 | 0.018503692 | 1.1028003 | 0 | 0.098158815 | 0.098158815 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spatial_with_i_only_mixed | 0.013759224 | 0.060612169 | 0.018503692 | 1.1028003 | 0 | 0.098158815 | 0.098158815 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spatial_with_i_etaeta_plus_mixed_no_AA | 0.013759224 | 0.060612169 | 0.018503692 | 1.1028003 | 0 | 0.098158815 | 0.098158815 | 0 | collective_total | spatial | with_i |
| 24 | (0.5235988,0.5235988) | pkg_total_spatial_with_i_AA_plus_mixed_no_etaeta | 0.0082065804 | 0.10162289 | 0.031023452 | 1.1028003 | 0.010604769 | 0.098158815 | 0.098158815 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0) | pkg_total_spatial_no_i | 0.017482345 | 0.036389577 | 0.032810139 | 1.2299211 | 0.027355675 | 0.11938487 | 0.11938487 | 0 | collective_total | spatial | no_i |
| 36 | (0.3490659,0) | pkg_total_spatial_with_i | 0.016834591 | 0.037789759 | 0.034072593 | 1.2299211 | 0.027355675 | 0.11938487 | 0.11938487 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0) | pkg_total_spacetime_omega_plus_no_i | 0.017475098 | 0.036404669 | 0.032823747 | 1.2299211 | 0.027333243 | 0.11933591 | 0.11933591 | 0 | collective_total | spacetime | no_i |
| 36 | (0.3490659,0) | pkg_total_spacetime_omega_plus_with_i | 0.016827339 | 0.037806046 | 0.034087278 | 1.2299211 | 0.027333243 | 0.11933591 | 0.11933591 | 0 | collective_total | spacetime | with_i |
| 36 | (0.3490659,0) | pkg_total_spacetime_omega_minus_with_i | 0.016856027 | 0.037741702 | 0.034029263 | 1.2299211 | 0.02742312 | 0.11953195 | 0.11953195 | 0 | collective_total | spacetime | with_i |
| 36 | (0.3490659,0) | pkg_total_spacetime_omega_zero_with_i | 0.016841671 | 0.037773873 | 0.03405827 | 1.2299211 | 0.027378126 | 0.11943385 | 0.11943385 | 0 | collective_total | spacetime | with_i |
| 36 | (0.3490659,0) | pkg_bubble_spatial_with_i | 0.0067920445 | 0.09366475 | 0.084451476 | 1.0627607 | 0.1065232 | 0.46480111 | 0.46480111 | 3.6046392 | bubble | spatial | with_i |
| 36 | (0.3490659,0) | pkg_counterterm_spatial_with_i | 0.010477398 | 0.060718813 | 0.054746245 | 1 | 0.13387888 | 0.5841568 | 0.5841568 | 2.7847358 | counterterm | spatial | with_i |
| 36 | (0.3490659,0) | pkg_bond_metric_spatial_with_i | 57.948612 | 1.0978264e-05 | 9.8983937e-06 | 68279.549 | 0.1318604 | 0.81676298 | 0.81676298 | 4.5163186 | bond_metric | spatial | with_i |
| 36 | (0.3490659,0) | pkg_hybrid_bond_metric_spatial_with_i | 0.0084659178 | 0.075145444 | 0.067753809 | 1.5703706 | 0.025337204 | 0.59247983 | 0.58785425 | 3.5774644 | hybrid | spatial | with_i |
| 36 | (0.3490659,0) | pkg_total_spatial_with_i_no_AA | 0.026383524 | 0.024112592 | 0.021740772 | 1.2299211 | 0 | 0.11938487 | 0.11938487 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0) | pkg_total_spatial_with_i_only_mixed | 0.026383524 | 0.024112592 | 0.021740772 | 1.2299211 | 0 | 0.11938487 | 0.11938487 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0) | pkg_total_spatial_with_i_etaeta_plus_mixed_no_AA | 0.026383524 | 0.024112592 | 0.021740772 | 1.2299211 | 0 | 0.11938487 | 0.11938487 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0) | pkg_total_spatial_with_i_AA_plus_mixed_no_etaeta | 0.016834591 | 0.037789759 | 0.034072593 | 1.2299211 | 0.027355675 | 0.11938487 | 0.11938487 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spatial_no_i | 0.012789378 | 0.049333788 | 0.020832801 | 1.0045471 | 0.020369192 | 0.12569644 | 0.12569644 | 0 | collective_total | spatial | no_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spatial_with_i | 0.012142857 | 0.051960464 | 0.021942 | 1.0045471 | 0.020369192 | 0.12569644 | 0.12569644 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spacetime_omega_plus_no_i | 0.012786754 | 0.049343913 | 0.020837076 | 1.0045471 | 0.020360837 | 0.12567066 | 0.12567066 | 0 | collective_total | spacetime | no_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spacetime_omega_plus_with_i | 0.012140229 | 0.05197171 | 0.021946749 | 1.0045471 | 0.020360837 | 0.12567066 | 0.12567066 | 0 | collective_total | spacetime | with_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spacetime_omega_minus_with_i | 0.012150715 | 0.05192686 | 0.02192781 | 1.0045471 | 0.020394285 | 0.12577384 | 0.12577384 | 0 | collective_total | spacetime | with_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spacetime_omega_zero_with_i | 0.01214547 | 0.051949285 | 0.021937279 | 1.0045471 | 0.02037755 | 0.12572223 | 0.12572223 | 0 | collective_total | spacetime | with_i |
| 36 | (0.3490659,0.3490659) | pkg_bubble_spatial_with_i | 0.0038960603 | 0.16194526 | 0.068386665 | 1.0019799 | 0.046570246 | 0.28737178 | 0.28737178 | 3.6046392 | bubble | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_counterterm_spatial_with_i | 0.0081515479 | 0.077402291 | 0.032685641 | 1 | 0.066939438 | 0.41306123 | 0.41306123 | 2.5102757 | counterterm | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_bond_metric_spatial_with_i | 0.0061072039 | 0.10331217 | 0.043626932 | 3.9670266e+15 | 0.064920967 | 0.57092895 | 0.57092895 | 4.3169941 | bond_metric | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_hybrid_bond_metric_spatial_with_i | 0.0058470681 | 0.10790852 | 0.045567893 | 1.8286928 | 0.018350721 | 0.42331813 | 0.42118719 | 3.5507086 | hybrid | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spatial_with_i_no_AA | 0.019253046 | 0.032771359 | 0.013838775 | 1.0045471 | 0 | 0.12569644 | 0.12569644 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spatial_with_i_only_mixed | 0.019253046 | 0.032771359 | 0.013838775 | 1.0045471 | 0 | 0.12569644 | 0.12569644 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spatial_with_i_etaeta_plus_mixed_no_AA | 0.019253046 | 0.032771359 | 0.013838775 | 1.0045471 | 0 | 0.12569644 | 0.12569644 | 0 | collective_total | spatial | with_i |
| 36 | (0.3490659,0.3490659) | pkg_total_spatial_with_i_AA_plus_mixed_no_etaeta | 0.012142857 | 0.051960464 | 0.021942 | 1.0045471 | 0.020369192 | 0.12569644 | 0.12569644 | 0 | collective_total | spatial | with_i |

## Human-readable conclusion

The LSQ mixed block still closes dwave in the quick cases, but remains only a diagnostic reference.
StageSC-2j longitudinal completion remains a partial-improvement reference rather than a closed solution.
Best package variants across dwave cases: {'pkg_bubble_spatial_with_i': 4}.
Best C(Q) candidates across dwave cases: {'bubble': 4}.
Best lambda conventions across dwave cases: {'spatial': 4}.
Best alpha conventions across dwave cases: {'with_i': 4}.
All controls monitored: ['dwave_const_form', 'onsite_s', 'spm'].
The audit tests whether AA, A-eta, eta-A, and eta-eta should be treated as one gauge-covariant package; it does not claim a full transverse microscopic current.
Formal Casimir input remains forbidden.
The result is suitable for the next analytic-design stage only as diagnostic evidence, not as a production implementation.

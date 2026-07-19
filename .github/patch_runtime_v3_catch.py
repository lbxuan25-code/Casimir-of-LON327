from pathlib import Path

path = Path("src/lno327/casimir/adaptive_joint_q.py")
text = path.read_text(encoding="utf-8")
old = '''    except RuntimeError as exc:
        reason = (
            "joint_microscopic_q_node_budget_exhausted"
            if str(exc) == "joint_microscopic_q_node_budget_exhausted"
            else f"joint_runtime_failure: {exc}"
        )
        return _unresolved_result(
            config,
            direction_records=direction_records,
            radial_run_records=radial_run_records,
            offset_record=None,
            selected_order=config.angular_orders[current_index],
            radial_round_cap=radial_round_cap,
            pairing_results=last_pairing_results,
            radial_passed=last_radial_passed,
            angular_passed=last_angular_passed,
            offset_passed=last_offset_passed,
            all_certified=all_certified,
            reason=reason,
            provider=active_provider,
        )
    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:
        return _unresolved_result(
            config,
            direction_records=direction_records,
            radial_run_records=radial_run_records,
            offset_record=None,
            selected_order=config.angular_orders[current_index],
            radial_round_cap=radial_round_cap,
            pairing_results=last_pairing_results,
            radial_passed=False,
            angular_passed=False,
            offset_passed=False,
            all_certified=False,
            reason=f"point_provider_failure: {exc}",
            provider=active_provider,
        )
'''
new = '''    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:
        return _unresolved_result(
            config,
            direction_records=direction_records,
            radial_run_records=radial_run_records,
            offset_record=None,
            selected_order=config.angular_orders[current_index],
            radial_round_cap=radial_round_cap,
            pairing_results=last_pairing_results,
            radial_passed=False,
            angular_passed=False,
            offset_passed=False,
            all_certified=False,
            reason=f"point_provider_failure: {exc}",
            provider=active_provider,
        )
    except RuntimeError as exc:
        reason = (
            "joint_microscopic_q_node_budget_exhausted"
            if str(exc) == "joint_microscopic_q_node_budget_exhausted"
            else f"joint_runtime_failure: {exc}"
        )
        return _unresolved_result(
            config,
            direction_records=direction_records,
            radial_run_records=radial_run_records,
            offset_record=None,
            selected_order=config.angular_orders[current_index],
            radial_round_cap=radial_round_cap,
            pairing_results=last_pairing_results,
            radial_passed=last_radial_passed,
            angular_passed=last_angular_passed,
            offset_passed=last_offset_passed,
            all_certified=all_certified,
            reason=reason,
            provider=active_provider,
        )
'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one catch-order replacement, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

def choose_port_conflict_action(
    *,
    requested_force: bool,
    owned_pids: set[int],
    listeners_by_port: dict[int, set[int]],
) -> str:
    all_listeners = set().union(*listeners_by_port.values()) if listeners_by_port else set()
    foreign = {pid for pid in all_listeners if pid not in owned_pids}
    if not all_listeners:
        return "ok"
    if foreign:
        return "force" if requested_force else "error"
    return "restart"

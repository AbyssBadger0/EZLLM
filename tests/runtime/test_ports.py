from ezllm.runtime.ports import choose_port_conflict_action


def test_choose_port_conflict_action_returns_ok_when_no_listeners():
    action = choose_port_conflict_action(
        requested_force=False,
        owned_pids={111},
        listeners_by_port={},
    )

    assert action == "ok"


def test_choose_port_conflict_action_restarts_when_all_listeners_are_owned():
    action = choose_port_conflict_action(
        requested_force=False,
        owned_pids={111, 222},
        listeners_by_port={8888: {111}, 8889: {222}},
    )

    assert action == "restart"


def test_choose_port_conflict_action_rejects_foreign_process_without_force():
    action = choose_port_conflict_action(
        requested_force=False,
        owned_pids={111},
        listeners_by_port={8888: {222}},
    )

    assert action == "error"


def test_choose_port_conflict_action_forces_foreign_process_when_requested():
    action = choose_port_conflict_action(
        requested_force=True,
        owned_pids={111},
        listeners_by_port={8888: {222}},
    )

    assert action == "force"

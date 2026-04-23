from ezllm.runtime.ports import choose_port_conflict_action


def test_choose_port_conflict_action_rejects_foreign_process_without_force():
    action = choose_port_conflict_action(
        requested_force=False,
        owned_pids={111},
        listeners_by_port={8888: {222}},
    )

    assert action == "error"

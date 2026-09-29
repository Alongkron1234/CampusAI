from campusai.cli import build_parser, main


def test_help_lists_all_commands(capsys):
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._subparsers._group_actions if action.dest == "command"
    )
    expected = {"check", "ingest", "index", "search", "ask", "chat", "eval"}
    assert set(subparsers_action.choices.keys()) == expected


def test_eval_refuses_locked_test_set_without_flag(capsys):
    from campusai import config

    for target in ("retrieval", "answers"):
        exit_code = main(["eval", target, "--questions", str(config.EVAL_TEST_QUESTIONS_PATH)])
        assert exit_code == 1
        assert "--allow-test-set" in capsys.readouterr().out

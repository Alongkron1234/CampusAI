from campusai.cli import build_parser, main


def test_help_lists_all_commands(capsys):
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._subparsers._group_actions if action.dest == "command"
    )
    expected = {"check", "ingest", "index", "search", "ask", "chat", "eval"}
    assert set(subparsers_action.choices.keys()) == expected


def test_unimplemented_command_returns_nonzero(capsys):
    # "check" implement แล้วตั้งแต่ Issue #2 จึงใช้ "ingest" (ยังไม่ implement) แทน
    exit_code = main(["ingest"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ingest" in captured.out

from campusai.cli import build_parser, main


def test_help_lists_all_commands(capsys):
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._subparsers._group_actions if action.dest == "command"
    )
    expected = {"check", "ingest", "index", "remove", "serve", "search", "ask", "chat", "eval"}
    assert set(subparsers_action.choices.keys()) == expected


def test_unimplemented_command_returns_nonzero(capsys):
    # ทุกคำสั่ง implement แล้ว เหลือ "eval answers" (Issue #8)
    exit_code = main(["eval", "answers"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "answers" in captured.out

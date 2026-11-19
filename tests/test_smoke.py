"""These pass on day one. They check the plumbing so you notice quickly if you break it."""


def test_the_command_line_runs(capsys):
    from app.cli import main

    main(["hello"])
    assert "is ready" in capsys.readouterr().out

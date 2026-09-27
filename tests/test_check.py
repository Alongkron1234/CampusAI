from campusai import config
from campusai.cli import main


def test_check_returns_error_when_no_pdf_found(tmp_path, monkeypatch, capsys):
    # ชี้ RAW_DIR ไปที่โฟลเดอร์ว่างชั่วคราว แทนที่จะพึ่งไฟล์ PDF จริงใน data/raw/
    # (ไม่ commit PDF ขึ้น git ดังนั้น CI ต้องไม่พึ่งไฟล์จริง)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)

    exit_code = main(["check"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ไม่พบไฟล์ PDF" in captured.out

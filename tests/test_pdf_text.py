from campusai.ingest.pdf_text import needs_ocr

GOOD_THAI_TEXT = "ข้อ 15 นักศึกษาต้องแต่งกายให้ถูกต้องตามระเบียบของมหาวิทยาลัยทุกครั้งที่เข้าสอบ"


def test_good_thai_text_does_not_need_ocr():
    ocr, reason = needs_ocr(GOOD_THAI_TEXT)
    assert ocr is False
    assert reason == "ข้อความใช้ได้"


def test_empty_text_needs_ocr():
    ocr, reason = needs_ocr("")
    assert ocr is True
    assert "สั้นเกินไป" in reason


def test_short_text_needs_ocr():
    ocr, reason = needs_ocr("   \n  ")
    assert ocr is True
    assert "สั้นเกินไป" in reason


def test_cid_marker_needs_ocr():
    text = "(cid:12)(cid:34)(cid:56) " * 5
    ocr, reason = needs_ocr(text)
    assert ocr is True
    assert "(cid:" in reason


def test_private_use_area_needs_ocr():
    pua_text = "" * 10
    ocr, reason = needs_ocr(pua_text)
    assert ocr is True
    assert "Private Use Area" in reason


def test_garbled_ascii_thai_document_needs_ocr():
    # จำลองเคสจริงจาก discipline2566.pdf: ดึงข้อความได้ยาวพอ แต่เป็นฟอนต์เพี้ยน (ASCII ล้วน)
    garbled = "uil?a AU U AV ?uuuaun'l::nu']2uu {v-ouaY AY o VY uvdr'lnfinutrav" * 3
    ocr, reason = needs_ocr(garbled)
    assert ocr is True
    assert "สัดส่วนอักษรไทยต่ำ" in reason


def test_mixed_thai_and_english_within_threshold_does_not_need_ocr():
    # มีคำอังกฤษปนบ้าง (เช่น "code of Honor") แต่ยังเป็นข้อความไทยเป็นหลัก
    mixed = GOOD_THAI_TEXT + " (Code of Honor) " + GOOD_THAI_TEXT
    ocr, reason = needs_ocr(mixed)
    assert ocr is False

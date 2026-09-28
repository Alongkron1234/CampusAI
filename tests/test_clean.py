from campusai.ingest.clean import (
    clean_text,
    collapse_blank_lines,
    fix_sara_am,
    flatten_html_tables,
    normalize_quotes,
    remove_figure_placeholders,
    remove_page_numbers,
    strip_trailing_whitespace,
    thai_digits_to_arabic,
)


def test_fix_sara_am_joins_nikhahit_and_sara_aa():
    # "นํา" ที่นิคหิต (ํ) แยกกับสระอา (า) ต้องกลายเป็น "นำ" ตัวเดียว
    broken = "นิคหิตแยก: นํา"
    assert fix_sara_am(broken) == "นิคหิตแยก: นำ"


def test_fix_sara_am_does_not_touch_already_correct_text():
    correct = "คำว่า นำ เขียนถูกอยู่แล้ว"
    assert fix_sara_am(correct) == correct


def test_thai_digits_to_arabic_converts_all_digits():
    assert thai_digits_to_arabic("ข้อ ๑๕ และ ๒๐๒๕") == "ข้อ 15 และ 2025"


def test_thai_digits_to_arabic_leaves_arabic_digits_unchanged():
    assert thai_digits_to_arabic("ข้อ 15") == "ข้อ 15"


def test_normalize_quotes_converts_curly_to_straight():
    mixed = "“มหาวิทยาลัย” และ ‘สภา’"
    assert normalize_quotes(mixed) == '"มหาวิทยาลัย" และ \'สภา\''


def test_normalize_quotes_handles_mismatched_open_close():
    # เคสจริงจาก ruleG2568.pdf: เปิดด้วยเครื่องหมายตรง ปิดด้วยโค้ง
    mismatched = '"มหาวิทยาลัย” หมายความว่า...'
    assert normalize_quotes(mismatched) == '"มหาวิทยาลัย" หมายความว่า...'


def test_flatten_html_tables_puts_clause_heading_back_at_line_start():
    # เคสจริงจาก ruleG2568.pdf: Typhoon OCR แปลงรายการนิยามศัพท์เป็นตาราง HTML
    # ทำให้ "ข้อ 4" ไม่ได้อยู่ต้นบรรทัด ต้องแตกตารางกลับเป็นบรรทัดปกติก่อน
    html_table = (
        "ก่อนหน้า\n"
        "<table><tr><td>ข้อ 4 ในระเบียบนี้</td><td></td></tr>"
        '<tr><td>"มหาวิทยาลัย"</td><td>หมายความว่า มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี</td></tr>'
        "</table>\n"
        "ต่อไป"
    )
    result = flatten_html_tables(html_table)

    assert "<table>" not in result
    assert "<td>" not in result
    lines = [line for line in result.split("\n") if line.strip()]
    assert lines[1] == "ข้อ 4 ในระเบียบนี้"
    assert lines[2] == '"มหาวิทยาลัย" หมายความว่า มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี'


def test_flatten_html_tables_drops_empty_cells():
    html_table = "<table><tr><td>หัวข้อ</td><td></td></tr></table>"
    result = flatten_html_tables(html_table)
    assert result.strip() == "หัวข้อ"


def test_flatten_html_tables_leaves_text_without_tables_unchanged():
    text = "ข้อ 1 ไม่มีตารางอยู่ในนี้เลย"
    assert flatten_html_tables(text) == text


def test_remove_figure_placeholders_removes_english_caption_line():
    text = "ข้อ 4 ในระเบียบนี้\n\n(Handwritten signature)\n\nข้อ 5 ต่อไป"
    result = remove_figure_placeholders(text)
    assert "(Handwritten signature)" not in result
    assert "ข้อ 4" in result
    assert "ข้อ 5" in result


def test_remove_figure_placeholders_keeps_thai_content_in_parens():
    # วงเล็บที่มีเนื้อหาไทยจริง (ไม่ใช่คำบรรยายรูป) ต้องไม่ถูกลบ
    text = "อาศัยอำนาจตามความในมาตรา 18 (2) แห่งพระราชบัญญัติ"
    assert remove_figure_placeholders(text) == text


def test_remove_page_numbers_removes_dash_style():
    text = "เนื้อหาบรรทัดก่อน\n- 3 -\nเนื้อหาบรรทัดถัดไป"
    result = remove_page_numbers(text)
    assert "- 3 -" not in result


def test_remove_page_numbers_removes_slash_style():
    text = "เนื้อหา\n3/10\nเนื้อหาต่อ"
    result = remove_page_numbers(text)
    assert "3/10" not in result


def test_remove_page_numbers_removes_word_style():
    text = "เนื้อหา\nหน้า 3\nเนื้อหาต่อ"
    result = remove_page_numbers(text)
    assert "หน้า 3" not in result


def test_collapse_blank_lines_reduces_to_double_newline():
    text = "ย่อหน้าแรก\n\n\n\n\nย่อหน้าสอง"
    assert collapse_blank_lines(text) == "ย่อหน้าแรก\n\nย่อหน้าสอง"


def test_strip_trailing_whitespace_removes_trailing_but_keeps_leading():
    text = "    3.1 ข้อย่อยที่เยื้อง   \nบรรทัดปกติ\t\t"
    result = strip_trailing_whitespace(text)
    assert result == "    3.1 ข้อย่อยที่เยื้อง\nบรรทัดปกติ"


def test_clean_text_end_to_end_matches_real_ocr_pattern():
    # จำลองข้อความจริงจาก Typhoon OCR ของ ruleG2568.pdf หน้า 1
    raw = (
        'ข้อ ๔ ในระเบียบนี้\n'
        '"มหาวิทยาลัย” หมายความว่า มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี\n'
        "\n\n\n"
        "(Handwritten signature)\n"
        "- 1 -\n"
    )
    result = clean_text(raw)

    assert "ข้อ 4 ในระเบียบนี้" in result
    assert '"มหาวิทยาลัย" หมายความว่า' in result
    assert "(Handwritten signature)" not in result
    assert "- 1 -" not in result
    assert "\n\n\n" not in result
    assert result == result.strip()  # ไม่มีช่องว่าง/บรรทัดว่างเกินหัว-ท้าย

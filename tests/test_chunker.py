from campusai.ingest.chunker import MAX_CHUNK_CHARS, chunk_document

DOC = "ruleG2568.pdf"


def test_normal_document_splits_one_chunk_per_clause():
    text = (
        "หมวด 1 บททั่วไป\n"
        "ข้อ 1 ระเบียบนี้เรียกว่า ระเบียบทดสอบ\n"
        "ข้อ 2 ระเบียบนี้ให้ใช้บังคับตั้งแต่วันประกาศ\n"
        "ข้อ 3 ให้ยกเลิกระเบียบเดิม\n"
    )
    chunks = chunk_document(text, DOC)

    assert [c.clause for c in chunks] == ["ข้อ 1", "ข้อ 2", "ข้อ 3"]
    assert all(c.chapter == "หมวด 1 บททั่วไป" for c in chunks)
    assert "ระเบียบนี้เรียกว่า" in chunks[0].text
    assert "ให้ยกเลิกระเบียบเดิม" in chunks[2].text


def test_chapter_changes_partway_through_document():
    text = (
        "หมวด 1 บททั่วไป\n"
        "ข้อ 1 เนื้อหาหมวดแรก\n"
        "หมวด 2 ระบบการจัดการศึกษา\n"
        "ข้อ 2 เนื้อหาหมวดสอง\n"
    )
    chunks = chunk_document(text, DOC)

    assert chunks[0].chapter == "หมวด 1 บททั่วไป"
    assert chunks[1].chapter == "หมวด 2 ระบบการจัดการศึกษา"


def test_preamble_before_first_clause_becomes_its_own_chunk():
    text = (
        "ระเบียบมหาวิทยาลัย ว่าด้วยเรื่องทดสอบ พ.ศ. 2568\n"
        "โดยเป็นการสมควรปรับปรุงระเบียบเดิม\n"
        "หมวด 1 บททั่วไป\n"
        "ข้อ 1 เนื้อหาข้อแรก\n"
    )
    chunks = chunk_document(text, DOC)

    assert chunks[0].clause is None
    assert "โดยเป็นการสมควร" in chunks[0].text
    assert chunks[1].clause == "ข้อ 1"


def test_nested_clause_number_like_5_slash_1():
    text = "ข้อ 5 เนื้อหาข้อ 5\nข้อ 5/1 เนื้อหาข้อ 5/1 ที่แทรกเพิ่มทีหลัง\nข้อ 6 เนื้อหาข้อ 6\n"
    chunks = chunk_document(text, DOC)

    assert [c.clause for c in chunks] == ["ข้อ 5", "ข้อ 5/1", "ข้อ 6"]
    assert "แทรกเพิ่มทีหลัง" in chunks[1].text


def test_false_reference_line_does_not_create_new_clause():
    # "ข้อ 3" ในบรรทัดนี้เป็นการอ้างอิงย้อนกลับ (เลขน้อยกว่าข้อก่อนหน้า) ไม่ใช่หัวข้อใหม่จริง
    text = (
        "ข้อ 14 เนื้อหาข้อ 14\n"
        "ข้อ 15 เนื้อหาข้อ 15 ซึ่งเป็นไปตาม\n"
        "ข้อ 3 แห่งระเบียบนี้ด้วย\n"
        "ข้อ 16 เนื้อหาข้อ 16\n"
    )
    chunks = chunk_document(text, DOC)

    assert [c.clause for c in chunks] == ["ข้อ 14", "ข้อ 15", "ข้อ 16"]
    # บรรทัดอ้างอิงลวงต้องถูกรวมเข้าเป็นเนื้อหาของข้อ 15 แทน ไม่หายไปไหน
    assert "ข้อ 3 แห่งระเบียบนี้ด้วย" in chunks[1].text


def test_long_clause_gets_split_with_overlap():
    long_body = "ก" * (MAX_CHUNK_CHARS + 500)
    text = f"ข้อ 1 {long_body}\nข้อ 2 เนื้อหาสั้น\n"

    chunks = chunk_document(text, DOC)
    clause_1_parts = [c for c in chunks if c.clause == "ข้อ 1"]

    assert len(clause_1_parts) == 2
    assert clause_1_parts[0].id.endswith("_1")
    assert clause_1_parts[1].id.endswith("_2")
    # ส่วนซ้อนทับ: ท้ายชิ้นแรกกับหัวชิ้นสองต้องมีข้อความซ้ำกันบางส่วน
    assert clause_1_parts[0].text[-100:] in clause_1_parts[1].text


def test_document_without_any_structure_falls_back_to_char_chunks():
    text = "เนื้อหาไม่มีโครงสร้างข้อ/มาตราเลย " * 100
    chunks = chunk_document(text, DOC)

    assert len(chunks) > 1
    assert all(c.clause is None and c.chapter is None for c in chunks)
    assert all(c.id.startswith("ruleG2568_chunk") for c in chunks)


def test_real_ocr_text_sample_from_ruleG2568():
    # ตัวอย่างจริงจาก Typhoon OCR หน้า 1 ของ ruleG2568.pdf (ผ่าน clean_text() มาแล้ว)
    real_text = (
        "ระเบียบมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี ว่าด้วย การศึกษาระดับบัณฑิตศึกษา พ.ศ. 2568\n\n"
        "โดยเป็นการสมควรที่จะปรับปรุงระเบียบมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี\n"
        "อาศัยอำนาจตามความในมาตรา 18 (2) แห่งพระราชบัญญัติมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี\n\n"
        "หมวด 1 บททั่วไป\n"
        'ข้อ 1 ระเบียบนี้เรียกว่า "ระเบียบมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี"\n'
        "ข้อ 2 ระเบียบนี้ให้ใช้บังคับตั้งแต่ภาคการศึกษาที่ 1 ปีการศึกษา 2568 เป็นต้นไป\n"
        "ข้อ 3 ให้ยกเลิก\n"
        "    3.1 ระเบียบเดิม พ.ศ. 2562\n"
        "    3.2 ระเบียบเดิม (ฉบับที่ 2) พ.ศ. 2563\n"
    )

    chunks = chunk_document(real_text, DOC)

    assert chunks[0].clause is None  # ส่วนเกริ่นนำ + อ้างอิงมาตรา 18 (2) ไม่ถูกตัดผิด
    assert "มาตรา 18 (2)" in chunks[0].text
    assert [c.clause for c in chunks[1:]] == ["ข้อ 1", "ข้อ 2", "ข้อ 3"]
    # ข้อย่อย 3.1 / 3.2 ต้องอยู่ในเนื้อหาของ "ข้อ 3" ไม่ใช่กลายเป็นหัวข้อแยก
    assert "3.1 ระเบียบเดิม" in chunks[3].text
    assert "3.2 ระเบียบเดิม" in chunks[3].text

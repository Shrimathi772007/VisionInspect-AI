from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.defect import Defect
from tests.conftest import make_image_bytes


def test_list_inspections(client, qe_headers, supervisor_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    created_id = upload_response.json()["id"]

    response = client.get("/inspections", headers=supervisor_headers)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert any(item["id"] == created_id for item in body)


def test_list_inspections_requires_authentication(client):
    response = client.get("/inspections")
    assert response.status_code == 401


def test_quality_engineer_can_upload_inspection(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["product_id"] == test_product["id"]
    assert body["status"] == "pending"
    assert body["source"] == "upload"
    assert "image_path" not in body


def test_factory_supervisor_cannot_upload(client, supervisor_headers, test_product):
    image_bytes = make_image_bytes("JPEG")
    response = client.post(
        "/inspections/upload",
        headers=supervisor_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 403


def test_upload_requires_authentication(client, test_product):
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 401


def test_upload_rejects_invalid_file_type(client, qe_headers, test_product):
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("not_an_image.txt", b"this is plain text, not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_fake_image_with_image_extension(client, qe_headers, test_product):
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("fake.png", b"not actually a png file", "image/png")},
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client, qe_headers, test_product):
    oversized_payload = b"\x00" * (11 * 1024 * 1024)
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("big.jpg", oversized_payload, "image/jpeg")},
    )
    assert response.status_code == 413


def test_upload_rejects_nonexistent_product(client, qe_headers):
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": "999999"},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 404


def test_get_inspection(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    inspection_id = upload_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}", headers=qe_headers)
    assert response.status_code == 200
    assert response.json()["id"] == inspection_id


def test_get_inspection_requires_authentication(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    inspection_id = upload_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}")
    assert response.status_code == 401


def test_get_inspection_image(client, qe_headers, supervisor_headers, test_product):
    image_bytes = make_image_bytes("PNG", size=(32, 32), color=(10, 200, 10))
    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    inspection_id = upload_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}/image", headers=supervisor_headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == image_bytes


def test_orphan_file_cleanup_on_db_failure(client, qe_headers, test_product, monkeypatch):
    from app.inspections import storage as storage_module

    original_save = storage_module.save_upload_file
    saved_paths = []

    async def spying_save_upload_file(product_id, upload_file):
        relative_path, absolute_path = await original_save(product_id, upload_file)
        saved_paths.append(absolute_path)
        return relative_path, absolute_path

    def failing_commit(self):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr("app.inspections.router.save_upload_file", spying_save_upload_file)
    monkeypatch.setattr(Session, "commit", failing_commit)

    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )

    assert response.status_code == 500
    assert len(saved_paths) == 1
    assert not saved_paths[0].exists(), "orphaned file was not cleaned up after DB failure"


def test_get_inspection_image_requires_authentication(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    inspection_id = upload_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}/image")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /inspections/{inspection_id}
# ---------------------------------------------------------------------------

def _create_upload_inspection(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_delete_inspection_requires_authentication(client, qe_headers, test_product):
    inspection_id = _create_upload_inspection(client, qe_headers, test_product)
    response = client.delete(f"/inspections/{inspection_id}")
    assert response.status_code == 401

    # inspection must still exist
    assert client.get(f"/inspections/{inspection_id}", headers=qe_headers).status_code == 200


def test_delete_inspection_requires_quality_engineer(client, qe_headers, supervisor_headers, test_product):
    inspection_id = _create_upload_inspection(client, qe_headers, test_product)
    response = client.delete(f"/inspections/{inspection_id}", headers=supervisor_headers)
    assert response.status_code == 403

    # inspection must still exist
    assert client.get(f"/inspections/{inspection_id}", headers=qe_headers).status_code == 200


def test_delete_nonexistent_inspection(client, qe_headers):
    response = client.delete("/inspections/999999", headers=qe_headers)
    assert response.status_code == 404


def test_delete_upload_inspection_removes_record_and_file(client, qe_headers, test_product):
    from app.inspections.storage import STORAGE_ROOT

    image_bytes = make_image_bytes("PNG")
    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert upload_response.status_code == 201
    inspection_id = upload_response.json()["id"]

    with SessionLocal() as session:
        from app.models.inspection import Inspection as InspectionModel

        image_path = session.get(InspectionModel, inspection_id).image_path
    absolute_path = STORAGE_ROOT / image_path
    assert absolute_path.is_file()

    response = client.delete(f"/inspections/{inspection_id}", headers=qe_headers)
    assert response.status_code == 204

    assert not absolute_path.exists(), "uploaded image file was not removed on delete"
    assert client.get(f"/inspections/{inspection_id}", headers=qe_headers).status_code == 404


def test_delete_inspection_cascades_defects(client, qe_headers, test_product):
    inspection_id = _create_upload_inspection(client, qe_headers, test_product)

    with SessionLocal() as session:
        defect = Defect(inspection_id=inspection_id, defect_type="scratch", confidence=0.9)
        session.add(defect)
        session.commit()
        defect_id = defect.id

    response = client.delete(f"/inspections/{inspection_id}", headers=qe_headers)
    assert response.status_code == 204

    with SessionLocal() as session:
        assert session.get(Defect, defect_id) is None


def test_delete_mvtec_inspection_does_not_touch_dataset_image(client, qe_headers, test_product):
    from app.inspections.storage import DATASET_ROOT

    on_disk_path = DATASET_ROOT / "bottle" / "test" / "good" / "000.png"
    original_bytes = on_disk_path.read_bytes()

    import_response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": "000.png",
        },
    )
    assert import_response.status_code == 201
    inspection_id = import_response.json()["id"]

    response = client.delete(f"/inspections/{inspection_id}", headers=qe_headers)
    assert response.status_code == 204

    assert on_disk_path.is_file(), "MVTec dataset image must never be deleted"
    assert on_disk_path.read_bytes() == original_bytes
    assert client.get(f"/inspections/{inspection_id}", headers=qe_headers).status_code == 404


def test_delete_inspection_rejects_path_traversal_in_stored_image_path(client, qe_headers, test_product, monkeypatch):
    inspection_id = _create_upload_inspection(client, qe_headers, test_product)

    with SessionLocal() as session:
        from app.models.inspection import Inspection as InspectionModel

        inspection = session.get(InspectionModel, inspection_id)
        inspection.image_path = "../../etc/passwd"
        session.commit()

    response = client.delete(f"/inspections/{inspection_id}", headers=qe_headers)
    assert response.status_code == 404

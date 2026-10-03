import json
import sys

sys.path.insert(0, r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import QE_EMAIL, QE_PASSWORD, register_or_login

client = TestClient(app)
creds = register_or_login(client, "Pytest QE", QE_EMAIL, QE_PASSWORD, "quality_engineer")
headers = {"Authorization": f"Bearer {creds['access_token']}"}

r = client.get("/inspections/analytics/summary", headers=headers)
print("default status:", r.status_code)
body = r.json()
print("top-level keys:", sorted(body.keys()))
print("performance:", json.dumps(body["performance"], indent=2))
print("trend period_days:", body["trend_monitoring"]["period_days"], "| activity days:", len(body["activity_by_day"]))

for days in (7, 14, 30):
    r = client.get(f"/inspections/analytics/summary?days={days}", headers=headers)
    b = r.json()
    print(f"days={days}: status={r.status_code} period_days={b['trend_monitoring']['period_days']} "
          f"trend_len={len(b['trend_monitoring']['daily'])} activity_len={len(b['activity_by_day'])} "
          f"perf.window={b['performance']['window_days']}")

for bad in ("5", "0", "-1", "abc", "31"):
    r = client.get(f"/inspections/analytics/summary?days={bad}", headers=headers)
    print(f"days={bad}: status={r.status_code}", r.json().get("detail") if r.status_code != 200 else "")

r = client.get("/inspections", headers=headers)
full = len(r.json())
r5 = client.get("/inspections?limit=5", headers=headers)
print("list all:", full, "| limit=5:", len(r5.json()), "| newest-first same head:", r5.json()[0]["id"] == r.json()[0]["id"])
for bad in ("0", "1001", "x"):
    print(f"limit={bad}:", client.get(f"/inspections?limit={bad}", headers=headers).status_code)

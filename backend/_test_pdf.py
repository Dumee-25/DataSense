import requests, json

r = requests.get("http://localhost:8000/api/jobs")
data = r.json()
jobs = data.get("jobs", [])
print("Total:", data.get("total", 0))
for j in jobs[:10]:
    sid = j.get("job_id") or j.get("id")
    print(f"  {sid} -> {j.get('status')} ({j.get('filename', '?')})")

# Try to find any completed job and download its PDF
completed = [j for j in jobs if j.get("status") == "completed"]
if completed:
    job_id = completed[0].get("job_id") or completed[0].get("id")
    print(f"\nTesting PDF for job: {job_id}")
    pdf_r = requests.get(f"http://localhost:8000/api/results/{job_id}/export/pdf")
    print(f"PDF status: {pdf_r.status_code}")
    if pdf_r.status_code != 200:
        print(f"Error: {pdf_r.text[:2000]}")
    else:
        print(f"PDF OK: {len(pdf_r.content)} bytes")
else:
    print("\nNo completed jobs found. Upload a CSV and try again.")

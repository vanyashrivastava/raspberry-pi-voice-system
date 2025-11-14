# How to Test Supabase via GitHub Actions

Follow these steps to query your data directly from GitHub and see it printed in the Actions workflow!

## Step 1: Update GitHub Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add three secrets:

| Secret Name | Value |
|---|---|
| `SUPABASE_URL` | `https://your_project_id.supabase.co` |
| `SUPABASE_KEY` | Your anon key from Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | Your service role key from Supabase |

> **Where to find these:** Supabase Dashboard → Settings → API

4. Click **Add secret** for each one

## Step 2: Trigger the Workflow

You have two options:

### Option A: Manual Trigger (Easiest)
1. Go to your repo → **Actions** tab
2. Find **Test Supabase Query** workflow
3. Click **Run workflow** button
4. Select the branch (main) and click the green **Run workflow** button
5. Watch it execute!

### Option B: Automatic Trigger
Push any commit to trigger the workflow:
```bash
git add .
git commit -m "Test Supabase"
git push origin main
```

## Step 3: Watch the Workflow Run

1. Go to **Actions** tab in GitHub
2. Click on the **Test Supabase Query** workflow run
3. Click on the job **query-supabase**
4. Click on the **Query Supabase Data** step to expand it
5. **Scroll down to see your data!** 👇

You'll see something like:

```
======================================================================
🔌 SUPABASE CONNECTION TEST
======================================================================

📍 Connecting to: https://your_project.supabase.co
✅ Connection successful!

📊 QUERYING TABLES
----------------------------------------------------------------------

📋 Table: call_transcripts
  Attempting to fetch data...
  ✅ Found 1 row(s)

  📌 ROW DATA:
  ────────────────────────────────────────────────────────
  {
    "id": 100,
    "created_at": "2025-11-14T04:16:04+00:00",
    "caller_id": "1234",
    "duration": 15,
    "transcript": "hello, my name is vanya and this is a test."
  }
  ────────────────────────────────────────────────────────

✅ SUPABASE TEST COMPLETE
======================================================================

If you saw your data above, the connection is working! 🎉
```

## What This Proves

✅ **GitHub can connect to Supabase** - Network connectivity works
✅ **Your credentials are correct** - Secrets are set properly
✅ **Your data exists in Supabase** - Tables have records
✅ **GitHub Actions can query your database** - CI/CD integration works
✅ **Everything is production-ready** - Ready to sync real data!

## Next Steps After Testing

1. **Update main.py** with Supabase integration (see `INTEGRATE_SUPABASE.md`)
2. **Push to GitHub** to trigger automated workflows
3. **Watch data sync** in the Actions tab
4. **View in Supabase Dashboard** to see data accumulate

## Troubleshooting

### Workflow Not Found
- Make sure `.github/workflows/test-query-supabase.yml` exists
- Go to Actions tab and refresh

### Workflow Failed
- Check the error message in the log
- Common issues:
  - **Missing secrets**: Verify SUPABASE_URL and SUPABASE_KEY are set
  - **Wrong credentials**: Double-check values from Supabase Dashboard
  - **Project paused**: Ensure Supabase project is active (not paused)

### No Data Shown
- The table exists but is empty
- You need to add at least one row of data
- Use Supabase Dashboard → Data Editor to add test data

### Connection Timeout
- Check your internet connection
- Verify Supabase project URL is correct
- Try again (may be temporary network issue)

## Running Locally First

Before relying on GitHub Actions, test locally:

```bash
# 1. Set up your credentials
python setup_supabase_credentials.py

# 2. Test connection locally
python test_supabase_query.py

# 3. If that works, GitHub Actions will work!
```

## Workflow File Explained

The workflow is defined in `.github/workflows/test-query-supabase.yml`:

```yaml
name: Test Supabase Query              # Workflow name (shows in Actions tab)

on:
  workflow_dispatch:                   # Can run manually
  push:
    branches: [ main ]                 # Runs on every push to main
```

The key part that queries your data:

```python
# Uses your GitHub Secrets
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Connects to Supabase
headers = {'Authorization': f'Bearer {SUPABASE_KEY}'}

# Queries each table
response = requests.get(
    f"{SUPABASE_URL}/rest/v1/call_transcripts?limit=1",
    headers=headers
)

# Prints the data
print(json.dumps(data, indent=2))
```

## Schedule It (Optional)

To run this automatically on a schedule (e.g., daily), edit `.github/workflows/test-query-supabase.yml`:

```yaml
on:
  schedule:
    - cron: '0 9 * * *'  # Run daily at 9 AM UTC
```

## Real-World Usage

Once you've verified it works with test data, the same workflow can:
- ✅ Query real transcripts from your audio
- ✅ Check latest alerts
- ✅ Monitor email detection results
- ✅ Track system metrics
- ✅ Send notifications if data looks wrong

## Learn More

- **INTEGRATE_SUPABASE.md** - How to add to main.py
- **SUPABASE_QUICK_REFERENCE.md** - SQL queries and commands
- **GET_STARTED.md** - Overall setup guide

---

**You're ready!** Push to GitHub and watch your data sync to the cloud! 🚀

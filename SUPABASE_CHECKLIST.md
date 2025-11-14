# Supabase Integration Checklist

Complete this checklist to fully set up Supabase integration with your GitHub repo.

## Phase 1: Supabase Setup ✓

- [ ] Create Supabase account at https://supabase.com
- [ ] Create new project named "raspberry-pi-scam-detection"
- [ ] Save project credentials (URL, keys) securely
- [ ] Copy credentials to `.env` file:
  ```bash
  cp .env.example .env
  # Edit .env with your actual credentials
  ```

## Phase 2: GitHub Setup ✓

- [ ] Add GitHub Secrets:
  - [ ] `SUPABASE_URL` 
  - [ ] `SUPABASE_KEY`
  - [ ] `SUPABASE_SERVICE_ROLE_KEY`
- [ ] Verify `.github/workflows/` folder exists
- [ ] GitHub Actions are enabled in your repository

## Phase 3: Database Schema ✓

- [ ] Open Supabase Dashboard → SQL Editor
- [ ] Create new query
- [ ] Copy entire content of `supabase/migrations/001_init_schema.sql`
- [ ] Paste into SQL Editor
- [ ] Click "Run"
- [ ] Verify tables created:
  - [ ] `audio_transcripts`
  - [ ] `alerts`
  - [ ] `email_events`
  - [ ] `system_logs`
  - [ ] `system_metrics`

## Phase 4: Local Testing ✓

- [ ] Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Test Supabase connection:
  ```bash
  python -c "from src.supabase.db_manager import SupabaseDB; db = SupabaseDB(); print('✓ Connected!' if db.test_connection() else '✗ Connection failed')"
  ```

- [ ] Run test script:
  ```bash
  python src/supabase/db_manager.py
  ```
  Should see test alerts/transcripts inserted

- [ ] Verify in Supabase Dashboard:
  - [ ] Go to Data Editor
  - [ ] Select `alerts` table
  - [ ] Confirm test data appears

## Phase 5: Code Integration ✓

Choose one or more integration approaches:

### Option A: SupabaseAlertLogger (Recommended)
- [ ] Update `main.py`:
  ```python
  from src.supabase.supabase_alert_logger import SupabaseAlertLogger
  alert_logger = SupabaseAlertLogger(use_supabase=True)
  ```

### Option B: Direct Database Access
- [ ] Integrate `SupabaseDB` into:
  - [ ] Audio stream handler
  - [ ] Email parser/inference engine
  - [ ] Alert components

### Option C: Batch Insert
- [ ] Use for high-volume data scenarios
- [ ] Call `db.batch_insert()` instead of individual inserts

## Phase 6: GitHub Actions ✓

- [ ] Verify workflows are set up:
  - [ ] `.github/workflows/supabase-sync.yml` exists
  - [ ] `.github/workflows/test-supabase.yml` exists

- [ ] Test workflow trigger:
  ```bash
  git add -A
  git commit -m "Add Supabase integration"
  git push origin main
  ```

- [ ] Monitor in GitHub:
  - [ ] Go to **Actions** tab
  - [ ] Watch for workflow runs
  - [ ] Verify they complete successfully

## Phase 7: Monitoring & Validation ✓

- [ ] Set up logging:
  - [ ] Verify system logs are captured:
    ```python
    db.insert_system_log('INFO', 'System started', component='main')
    ```

- [ ] Monitor performance:
  - [ ] Insert metrics:
    ```python
    db.insert_metric('cpu_usage', 45.2, unit='%', component='system')
    ```

- [ ] Check Supabase Dashboard:
  - [ ] Go to **Logs** tab
  - [ ] Look for any errors
  - [ ] Review sync activity

## Phase 8: Security ✓

- [ ] Review Row Level Security:
  - [ ] Go to **Authentication** → **Policies** in Supabase
  - [ ] Current policies allow public read
  - [ ] For production: implement stricter access control

- [ ] Verify `.env` is in `.gitignore`:
  ```bash
  echo ".env" >> .gitignore
  ```

- [ ] Never commit `.env` file to GitHub
- [ ] Use GitHub Secrets only for sensitive credentials

## Phase 9: Documentation ✓

- [ ] Read `SUPABASE_SETUP.md` for detailed guide
- [ ] Review `src/supabase/integration_examples.py` for code samples
- [ ] Share credentials with team members via secure channel
- [ ] Document any custom integrations

## Phase 10: Production Deployment ✓

- [ ] Test on Raspberry Pi:
  - [ ] Install dependencies on Pi
  - [ ] Test Supabase connectivity
  - [ ] Monitor network usage

- [ ] Enable Supabase backups:
  - [ ] Go to **Settings** → **Database**
  - [ ] Enable automatic backups

- [ ] Set up monitoring/alerts:
  - [ ] Define what "high alert" means for your use case
  - [ ] Consider Slack/email notifications for critical detections

- [ ] Performance optimization:
  - [ ] Use batch inserts for high-volume data
  - [ ] Monitor request rate vs limits
  - [ ] Adjust sync intervals as needed

## Verification Checklist

After completing all phases, verify:

```bash
# 1. Environment variables loaded
echo $SUPABASE_URL

# 2. Dependencies installed
python -c "import supabase; print('✓ supabase installed')"

# 3. Connection works
python src/supabase/db_manager.py

# 4. Tables exist
# (Check Supabase Dashboard > Data Editor)

# 5. Test data appears
# (Check audio_transcripts, alerts tables in Supabase)

# 6. GitHub Actions pass
# (Check Actions tab in GitHub)
```

## Troubleshooting

If something doesn't work:

1. **Connection Error**: Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
2. **Table Not Found**: Run migration in Supabase SQL Editor again
3. **Data Not Syncing**: Check GitHub Actions logs for errors
4. **Missing Imports**: Run `pip install -r requirements.txt` again
5. **Permission Denied**: Ensure proper file permissions on local log files

See **SUPABASE_SETUP.md** for detailed troubleshooting.

## Support & Resources

- 📚 [SUPABASE_SETUP.md](./SUPABASE_SETUP.md) - Complete setup guide
- 📝 [Integration Examples](./src/supabase/integration_examples.py)
- 🔌 [Database Manager Docs](./src/supabase/db_manager.py) - Inline docs
- 🌐 [Supabase Official Docs](https://supabase.com/docs)
- 📖 [PostgreSQL Docs](https://www.postgresql.org/docs/)

---

**Status**: Ready for deployment! ✅

Once you complete all checkboxes, your system will automatically:
- 📱 Sync audio transcripts with timestamps to Supabase
- 🚨 Log alerts to cloud database in real-time
- 📧 Store email analysis results
- 📊 Track system performance metrics
- 🔄 Backup data with redundancy

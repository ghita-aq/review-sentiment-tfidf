# How to Deploy Your Sentiment Analysis App (100% Free)

Your app is a **Django web application** with a pre-trained ML model. This guide shows you **two completely free** ways to deploy it so anyone can access it through a website URL.

---

## Option 1: Render.com (Recommended — Easiest)

**Render** gives you a free web service tier that's perfect for this project. No credit card needed.

### Step-by-Step

### Step 1: Push Your Code to GitHub

1. Go to [github.com](https://github.com) and create a free account if you don't have one.
2. Create a new repository (e.g., `review-sentiment-tfidf`).
3. On your computer, open a terminal in the project folder and run:

```powershell
cd "Z:\MQL\S3\Gen AI\review-sentiment-tfidf"

# Initialize git (skip if already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: sentiment analysis app"

# Link to your GitHub repo (replace YOUR_USERNAME and YOUR_REPO)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 2: Create a Free Render Account

1. Go to [render.com](https://render.com) and sign up with your GitHub account.
2. Click **"New +"** → **"Web Service"**.
3. Connect your GitHub repository and select `review-sentiment-tfidf`.

### Step 3: Configure the Web Service

Fill in these settings:

| Setting | Value |
|---|---|
| **Name** | `review-sentiment` (or any name you like) |
| **Region** | Choose the closest to you |
| **Branch** | `main` |
| **Root Directory** | `back-end-django` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn sentiment_app.wsgi --bind 0.0.0.0:$PORT` |
| **Type / Instance Type** | **Free** |

### Step 4: Set Environment Variables

In the Render dashboard, under **"Environment"** for your service, add:

| Key | Value |
|---|---|
| `DJANGO_DEBUG` | `False` |
| `DJANGO_SECRET_KEY` | Any random long string (e.g. `my-super-secret-key-12345-abcdef`) |

> **How to generate a secret key:** Run this in Python:
> ```python
> import secrets
> print(secrets.token_urlsafe(50))
> ```

### Step 5: Deploy

Click **"Create Web Service"**. Render will automatically build and deploy your app. This takes about 2–5 minutes.

Once deployed, your app will be live at:
```
https://review-sentiment-XXXX.onrender.com
```

> **Note:** Free Render instances "sleep" after 15 minutes of inactivity. The first request after sleeping takes ~30 seconds to wake up. This is normal.

---

## Option 2: PythonAnywhere (Alternative)

**PythonAnywhere** offers a free tier with Python and Django support.

### Step 1: Sign Up

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com) and create a **beginner (free)** account.

### Step 2: Upload Your Code

1. In PythonAnywhere, open a **Bash console**.
2. Clone your GitHub repo:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd review-sentiment-tfidf
```

### Step 3: Install Dependencies

```bash
cd back-end-django
pip3.10 install -r requirements.txt
```

### Step 4: Configure the Web App

1. Go to the **Web** tab in PythonAnywhere.
2. Click **"Add a new web app"**.
3. Choose **"Manual Config"** → **Python 3.10** (or latest available).
4. Under **"Code"** section:
   - **Source code:** `/home/YOUR_USERNAME/review-sentiment-tfidf/back-end-django`
   - **Working directory:** `/home/YOUR_USERNAME/review-sentiment-tfidf/back-end-django`
5. Edit the **WSGI configuration file** and replace with:
```python
import os
import sys

path = '/home/YOUR_USERNAME/review-sentiment-tfidf/back-end-django'
if path not in sys.path:
    sys.path.append(path)

os.environ['DJANGO_SETTINGS_MODULE'] = 'sentiment_app.settings'
os.environ['DJANGO_SECRET_KEY'] = 'my-super-secret-key-12345-abcdef'
os.environ['DJANGO_DEBUG'] = 'False'

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
```

6. Under **"Static files"**, map `/static/` to `/home/YOUR_USERNAME/review-sentiment-tfidf/back-end-django/staticfiles/`

### Step 5: Collect Static Files & Reload

In the Bash console:
```bash
cd /home/YOUR_USERNAME/review-sentiment-tfidf/back-end-django
python3.10 manage.py collectstatic --noinput
```

Then go back to the **Web** tab and click **"Reload"**.

Your app will be live at:
```
https://YOUR_USERNAME.pythonanywhere.com
```

---

## What Files Were Created for Deployment

I've added these files to make deployment easy:

| File | Purpose |
|---|---|
| `back-end-django/Procfile` | Tells Render how to start your app |
| `back-end-django/sentiment_app/settings.py` | Updated with `STATIC_ROOT` for production static files |
| `.gitignore` | Keeps build artifacts and caches out of Git |

---

## Troubleshooting

### "Model not loaded" error after deployment
The model files (`models/*.pkl`) must be in the `models/` folder at the project root (alongside `back-end-django/`). Make sure they're committed to Git:
```powershell
git add models/
git commit -m "Add model files"
git push
```

### App won't start on Render
Check the **Render logs** in your dashboard. Common issues:
- Missing `gunicorn` in `requirements.txt` — ✅ already included
- Wrong root directory — make sure it's set to `back-end-django`

### Static files not loading
Run `python manage.py collectstatic` locally and commit the `staticfiles/` folder, or make sure your deployment runs this command.

---

## Summary

| Platform | Free? | Custom Domain? | Sleeps? | Difficulty |
|---|---|---|---|---|
| **Render.com** | ✅ Yes | ❌ (paid) | ⚠️ After 15 min | ⭐ Easy |
| **PythonAnywhere** | ✅ Yes | ❌ (paid) | ❌ No sleep | ⭐⭐ Medium |

**My recommendation:** Start with **Render.com** — it's the easiest and takes about 10 minutes from start to finish.

---

## Need Help?

If you get stuck at any step, just ask! I can help you troubleshoot errors or walk through any step in more detail.

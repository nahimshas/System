# Sports Betting Analysis System
### NBA + MLB — Daily edge finder for Robinhood contracts

Runs automatically at 9am ET daily via GitHub Actions. Generates a mobile-friendly HTML report (GitHub Pages) and sends it to your email.

---

## Setup (one-time, ~15 minutes)

### Step 1 — Create a GitHub account
Go to [github.com](https://github.com) and sign up for a free account.

### Step 2 — Create a new GitHub repository
1. Click **New repository**
2. Name it `sports-betting-system` (or anything you want)
3. Set it to **Private**
4. Click **Create repository**

### Step 3 — Upload this code
Upload the contents of this folder to your new repo.
The easiest way: use [github.com/new](https://github.com) and drag-drop the files, or use GitHub Desktop.

### Step 4 — Get a free Odds API key
1. Go to [the-odds-api.com](https://the-odds-api.com) and sign up (free)
2. Copy your API key from the dashboard
3. Free tier = 500 credits/month (this system uses ~120/month)

### Step 5 — Add GitHub Secrets
In your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

| Secret Name | Value |
|---|---|
| `ODDS_API_KEY` | Your key from The Odds API |
| `EMAIL_PASSWORD` | Your Hotmail/Outlook password |
| `EMAIL_FROM` | nahimshas@hotmail.com |
| `EMAIL_TO` | nahimshas@hotmail.com (comma-separated for multiple) |

> **Tip for email**: If your Hotmail has 2-factor auth enabled, you need to create an App Password in Microsoft account security settings instead of using your regular password.

### Step 6 — Enable GitHub Pages
In your repo → **Settings** → **Pages**
- Source: **Deploy from a branch**
- Branch: **gh-pages** / **root**
- Click **Save**

Your mobile URL will be: `https://YOUR-GITHUB-USERNAME.github.io/sports-betting-system/`

### Step 7 — Enable GitHub Actions
In your repo → **Actions** → Click **"I understand my workflows, go ahead and enable them"**

### Step 8 — Run it manually to test
In your repo → **Actions** → **Daily Betting Report** → **Run workflow** → **Run workflow**

After ~2-3 minutes, check your GitHub Pages URL and your email.

---

## Manual Trigger (run anytime)

**Via GitHub (on your phone):**
1. Open your repo on GitHub
2. Tap **Actions** tab
3. Tap **Daily Betting Report**
4. Tap **Run workflow**
5. Optionally select a league (NBA only or MLB only)
6. Tap **Run workflow**

---

## Understanding the Report

### Singles Section
Each bet shows:
- **Market %** — what the sportsbook (DraftKings) implies the probability is
- **Model %** — what our statistical model calculates the true probability to be
- **Edge** — the difference. Minimum 3% required to recommend
- **Robinhood Action** — exact number of contracts to buy, total cost, and expected profit/loss

> The contract price shown is estimated from DraftKings odds. **Always verify the actual price on Robinhood before buying.**

### Parlays Section
2-leg parlays built from the top singles. Higher risk, higher reward. Each leg must have ≥2.5% individual edge. Budget allocation uses half-Kelly.

### Props Research Section
Statistical model picks for player props (strikeouts, team scoring, etc.). **No market price is shown here** — you need to check Robinhood's actual price for the prop and compare to the model line.

### Budget Allocation
Uses the **¼ Kelly Criterion** adapted for Robinhood's $0.02/contract commission:
- Bets with more edge get more allocation
- Reserve shows undeployed budget — do not force bets to use it
- Parlays use ½ of their calculated Kelly (higher variance)

---

## Adding More Leagues (Future)
To add NFL or NHL later:
1. Add sport key to `src/config.py`
2. Add stats fetcher in `src/data/`
3. Add analyzer in `src/models/edge_finder.py`
4. Wire up in `src/main.py`

---

## Daily Credit Usage (The Odds API)
| Call | Credits | Runs/day |
|---|---|---|
| NBA odds (h2h+spreads+totals) | 1 | 1 |
| MLB odds (h2h+spreads+totals) | 1 | 1 |
| **Total per day** | **~2–4** | |
| **Monthly total** | **~60–120** | |
| **Free tier limit** | **500** | |

# rajunekadi.com

Personal site, built with Jekyll and hosted on GitHub Pages.

## Everyday edits

| Want to…              | Edit this                                   |
|-----------------------|---------------------------------------------|
| Write a new post      | Add `_posts/YYYY-MM-DD-slug.md` (copy an existing one) |
| Add a project         | `_data/projects.yml`                        |
| Add a paper           | `_data/research.yml`                        |
| Add a talk            | `_data/talks.yml`                           |
| Change bio / intros   | `index.html`                                |
| Change the photo      | Save to `assets/img/raju.jpg`, uncomment the `<img>` line in `index.html` |
| Upload a CV           | Save as `assets/cv.pdf`                     |
| Colors / fonts        | `assets/css/style.css`                      |
| Email / social links  | `_config.yml`                               |

Every push to `main` rebuilds the site in about a minute. You can also edit any
file directly on github.com (pencil icon) and commit; no local tools needed.

## First-time publish

1. Create a **public** repo on GitHub named `rajunekadi.com`.
2. In this folder:
   ```
   git init
   git add .
   git commit -m "Initial site"
   git branch -M main
   git remote add origin https://github.com/rnekadi369/rajunekadi.com.git
   git push -u origin main
   ```
3. On GitHub: Settings → Pages → Source: "Deploy from a branch" → `main` / `/ (root)` → Save.
4. The site appears at `https://rnekadi369.github.io/rajunekadi.com/` within a few minutes.

## Custom domain (rajunekadi.com)

At your domain registrar's DNS settings add:

| Type  | Name | Value                     |
|-------|------|---------------------------|
| A     | @    | 185.199.108.153           |
| A     | @    | 185.199.109.153           |
| A     | @    | 185.199.110.153           |
| A     | @    | 185.199.111.153           |
| CNAME | www  | rnekadi369.github.io      |

Then on GitHub: Settings → Pages → Custom domain: `rajunekadi.com` → Save, wait for the
DNS check to pass, and tick **Enforce HTTPS**. The `CNAME` file in this repo keeps the
setting across deploys.

## Preview locally (optional)

```
gem install bundler
bundle install
bundle exec jekyll serve
```
Open http://localhost:4000.

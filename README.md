## Mail Fairy

Scans a maildir from a mailing list. She will then with GCLI post replies to pull request and issue
threads to a forge (Forgejo, Gitlab, Github, Gittea).
She detects replies by analyzing in-reply-to header trees and thus needs more than just 1 mail
She will check and cache all previously posted comments to avoid duplicates.
She will also check for "full quotes" and can be configured to remove these quotes or skip affected mail
mailman footers will be stripped, messages will be prefixed by author and date and should be a clickable
link to a mailman3 mailinglist (lore supported too)

Send pull request if something doesnt work or looks ugly or is aisloppy. Make sure its read/reviewable and tested!
(Github/Gitlab untested but GCLI supports them so they should work)

#### Example:

./mail_fairy.py \
    --maildir ~/mail/ffmpeg/dev \
    --owner FFmpeg --repo FFmpeg \
    --forge-base-url https://code.ffmpeg.org \
    --gcli-account mf \
    --max-age-days 14 \
    --manual  -v


## Supporte Forges
* GitHub (untested)
* GitLab (untested)
* Gitea  (untested)
* Forgejo

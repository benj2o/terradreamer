# Cleaning up lxhalle after a run

**Short answer: yes, clean up — but only after the merge has succeeded locally.**

`/var/tmp` is shared scratch on a machine with ~300 users, and the TUM banner
asks for it to be used responsibly. You are holding ~5.4 GB there.

---

## Do NOT clean up until this passes on your Mac

```bash
cd "/Users/benji/Code/NeurIPS CCAI 2026" && .venv/bin/python -c "
import pandas as pd
d = pd.read_csv('data/scaled_32UQC/p3_extreme_results.csv')
print('merged rows:', len(d))
print(d.fold_mode.value_counts().to_string())
assert len(d) == 1540 and set(d.fold_mode) == {'cube','loco','spatial_block'}
print('SAFE TO CLEAN THE SERVER')
"
```

Everything on the server is reproducible from your Mac, so this is not
destructive in principle — but re-staging costs a 4 GB transfer and two hours,
so confirm first.

---

## What is where

| path | size | keep? |
|---|---|---|
| `/var/tmp/$USER/p3/data/scaled_32UQC/{raw,embeddings,embeddings_cir,masks}` | ~4.1 GB | **delete** — identical copies on the Mac |
| `/var/tmp/$USER/p3/data/scaled_32UQC/p3_extreme_loco_*` | ~25 MB | **delete after merge** — retrieved already |
| `/var/tmp/$USER/p3/loco_console.txt` | small | delete — duplicate of the run log |
| `~/p3/.venv` | **1.3 GB** | see below |
| `~/p3` (code) | ~3 MB | keep, harmless |
| `~/p3_loco_handoff.tar.gz` | ~3 MB | delete after `scp` |

---

## Option A — full clean (you are done with the server)

```bash
tmux kill-session -t loco 2>/dev/null; rm -rf /var/tmp/$USER/p3 ~/p3 ~/p3_loco_handoff.tar.gz && echo "server cleaned" && df -h /var/tmp | tail -1
```

## Option B — keep the environment, free the data (RECOMMENDED)

If you may run more tiles, keeping the venv saves rebuilding it (~5 min and a
torch download). This frees the 4 GB that actually matters and leaves a
ready-to-go toolchain.

```bash
tmux kill-session -t loco 2>/dev/null; rm -rf /var/tmp/$USER/p3/data /var/tmp/$USER/p3/loco_console.txt ~/p3_loco_handoff.tar.gz && echo "data freed, venv + code kept" && du -sh ~/p3 && df -h /var/tmp | tail -1
```

Next tile then starts at step 5 of `SERVER_REPLICATION_STEPS.md`.

---

## Verify

```bash
echo "--- /var/tmp ---"; du -sh /var/tmp/$USER 2>/dev/null || echo "(gone)"; echo "--- home ---"; du -sh ~/p3 2>/dev/null || echo "(gone)"; echo "--- tmux ---"; tmux ls 2>&1; echo "--- processes ---"; pgrep -af run_p3_extreme || echo "(none)"
```

---

## Note

`/var/tmp` is swept periodically (Ubuntu's default is ~30 days untouched), so
anything left there disappears on its own eventually. That is a reason not to
*rely* on it, not a reason to skip cleaning up — leaving 4 GB parked for a
month on a shared box is exactly what the banner asks you not to do.

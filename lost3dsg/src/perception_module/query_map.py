"""
query_map.py
------------
Interroga la mappa temporale da terminale.
Non serve ROS2 attivo.

USO:
    python3 query_map.py --db /root/exchange/output/tiago_temporal_map.db
"""

import sys
import sqlite3
from pathlib import Path


class MapQuery:
    """Query dirette sul DB — indipendente da map_database.py."""

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser().resolve())
        if not Path(self.db_path).exists():
            print(f"[ERRORE] File non trovato: {self.db_path}")
            sys.exit(1)
        print(f"[MapDB] Database: {self.db_path}")

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _like(self, label: str) -> str:
        """'book' → '%book%'  così trova anche 'book#1', 'book#2' ecc."""
        return f"%{label.strip()}%"

    # ------------------------------------------------------------------ #

    def dove(self, label: str, color: str = "") -> list:
        """Posizione attuale (cerca con LIKE, include inattivi se nessun attivo)."""
        with self._conn() as conn:
            # Prima cerca attivi
            q = "SELECT * FROM objects WHERE label LIKE ?"
            params = [self._like(label)]
            if color:
                q += " AND color LIKE ?"
                params.append(self._like(color))
            q += " AND is_active=1 ORDER BY last_seen DESC"
            rows = conn.execute(q, params).fetchall()

            # Se nessun attivo, mostra anche gli inattivi
            if not rows:
                q2 = "SELECT * FROM objects WHERE label LIKE ?"
                params2 = [self._like(label)]
                if color:
                    q2 += " AND color LIKE ?"
                    params2.append(self._like(color))
                q2 += " ORDER BY last_seen DESC LIMIT 5"
                rows = conn.execute(q2, params2).fetchall()

            return [dict(r) for r in rows]

    def storia(self, label: str, color: str = "", n: int = 10) -> list:
        """Storico eventi di un oggetto — cerca direttamente in object_history."""
        with self._conn() as conn:
            q = """SELECT label, color, timestamp, event_type, phase, step,
                          x_old, y_old, z_old, x_new, y_new, z_new,
                          distance, iou, notes
                   FROM object_history
                   WHERE label LIKE ?"""
            params = [self._like(label)]
            if color:
                q += " AND color LIKE ?"
                params.append(self._like(color))
            q += " ORDER BY timestamp DESC LIMIT ?"
            params.append(n)
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    def spostato(self, label: str, color: str = "") -> list:
        """Cerca eventi 'moved' direttamente in object_history con LIKE."""
        with self._conn() as conn:
            q = """SELECT label, color, COUNT(*) as volte,
                          MIN(timestamp) as prima,
                          MAX(timestamp) as ultima,
                          AVG(distance) as dist_media
                   FROM object_history
                   WHERE event_type='moved' AND label LIKE ?"""
            params = [self._like(label)]
            if color:
                q += " AND color LIKE ?"
                params.append(self._like(color))
            q += " GROUP BY label, color"
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    def quanti(self):
        with self._conn() as conn:
            attivi = conn.execute("SELECT COUNT(*) FROM objects WHERE is_active=1").fetchone()[0]
            totale = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
            return attivi, totale

    def lista(self, only_active=True) -> list:
        with self._conn() as conn:
            q = "SELECT * FROM objects"
            if only_active:
                q += " WHERE is_active=1"
            q += " ORDER BY label, color"
            return [dict(r) for r in conn.execute(q).fetchall()]

    def labels(self) -> list:
        """Tutti i label distinti presenti nel DB."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT label, color, is_active FROM objects ORDER BY label"
            ).fetchall()
            return [dict(r) for r in rows]


# ------------------------------------------------------------------ #
#  HELPERS DI STAMPA                                                   #
# ------------------------------------------------------------------ #

def pos(x, y, z):
    try:
        return f"({float(x):.2f}, {float(y):.2f}, {float(z):.2f})"
    except (TypeError, ValueError):
        return "(null)"


def main():
    # Percorso DB
    db_path = "/root/exchange/output/tiago_temporal_map_1.db"
    if "--db" in sys.argv:
        db_path = sys.argv[sys.argv.index("--db") + 1]

    db = MapQuery(db_path)

    print("\n🤖  QUERY MAPPA TEMPORALE TIAGO")
    print("─" * 50)
    print("  dove <label> [colore]     posizione attuale")
    print("  storia <label> [colore]   storico eventi")
    print("  spostato <label>          si è mai mosso?")
    print("  quanti                    conta oggetti")
    print("  lista                     oggetti attivi")
    print("  tutto                     tutti (anche rimossi)")
    print("  labels                    tutti i label nel DB")
    print("  exit                      esci")
    print("─" * 50)
    print("  💡 label parziale: 'book' trova 'book#1', 'book#2'")
    print("─" * 50 + "\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()
        label = parts[1] if len(parts) > 1 else ""
        color = parts[2] if len(parts) > 2 else ""

        # ── EXIT ──────────────────────────────────────────
        if cmd == "exit":
            break

        # ── DOVE ──────────────────────────────────────────
        elif cmd == "dove" and label:
            results = db.dove(label, color)
            if results:
                for obj in results:
                    stato  = "ATTIVO" if obj["is_active"] else "RIMOSSO"
                    unc    = "  ⚠️ UNCERTAIN" if obj["is_uncertain"] else ""
                    print(f"  📍 [{stato}] {obj['color']} {obj['label']}  "
                          f"{pos(obj['x'], obj['y'], obj['z'])}"
                          f"  evento={obj['last_event']}"
                          f"  visto={obj['last_seen'][11:16]}{unc}")
            else:
                print(f"  ❓ Nessun oggetto con label '{label}' nel DB.")

        # ── STORIA ────────────────────────────────────────
        elif cmd == "storia" and label:
            history = db.storia(label, color)
            if history:
                print(f"  📜 Storico '{label}':")
                for h in history:
                    ts  = h["timestamp"][11:16]
                    ev  = h["event_type"]
                    lbl = f"{h['color']} {h['label']}"
                    if ev == "detected":
                        print(f"    [{ts}] ✅ RILEVATO   {lbl}  @ {pos(h['x_new'], h['y_new'], h['z_new'])}"
                              f"  [{h['phase']} step {h['step']}]")
                    elif ev == "moved":
                        print(f"    [{ts}] 🔄 SPOSTATO  {lbl}  "
                              f"{pos(h['x_old'], h['y_old'], h['z_old'])} → "
                              f"{pos(h['x_new'], h['y_new'], h['z_new'])}"
                              f"  Δ={float(h['distance']):.2f}m  IoU={float(h['iou']):.2f}")
                    elif ev == "disappeared":
                        print(f"    [{ts}] ❌ SCOMPARSO {lbl}  "
                              f"da {pos(h['x_old'], h['y_old'], h['z_old'])}"
                              f"  [{h['phase']} step {h['step']}]")
                    elif ev == "uncertain_added":
                        print(f"    [{ts}] ⚠️  UNCERTAIN  {lbl}")
            else:
                print(f"  ❓ Nessuna storia per '{label}' nel DB.")

        # ── SPOSTATO ──────────────────────────────────────
        elif cmd == "spostato" and label:
            results = db.spostato(label, color)
            if results:
                for r in results:
                    print(f"  ✅ SÌ — {r['color']} {r['label']}  "
                          f"spostato {r['volte']} volta/e  "
                          f"dist media={float(r['dist_media']):.2f}m  "
                          f"prima={r['prima'][11:16]}  ultima={r['ultima'][11:16]}")
            else:
                print(f"  ❌ NO — '{label}' non si è mai spostato (o non trovato)")

        # ── QUANTI ────────────────────────────────────────
        elif cmd == "quanti":
            attivi, totale = db.quanti()
            print(f"  📦 Attivi: {attivi}  |  Totale storico: {totale}")

        # ── LISTA ─────────────────────────────────────────
        elif cmd == "lista":
            objs = db.lista(only_active=True)
            print(f"\n  {'='*50}")
            print(f"  OGGETTI ATTIVI — {len(objs)}")
            print(f"  {'='*50}")
            for o in objs:
                unc = "  ⚠️" if o["is_uncertain"] else ""
                print(f"  {o['color']:8s} {o['label']:18s} "
                      f"{pos(o['x'], o['y'], o['z'])}"
                      f"  [{o['last_event']}]{unc}")
            print(f"  {'='*50}\n")

        # ── TUTTO ─────────────────────────────────────────
        elif cmd == "tutto":
            objs = db.lista(only_active=False)
            print(f"\n  {'='*50}")
            print(f"  TUTTI GLI OGGETTI — {len(objs)}")
            print(f"  {'='*50}")
            for o in objs:
                stato = "✅" if o["is_active"] else "❌"
                print(f"  {stato} {o['color']:8s} {o['label']:18s} "
                      f"{pos(o['x'], o['y'], o['z'])}"
                      f"  [{o['last_event']}]")
            print(f"  {'='*50}\n")

        # ── LABELS ────────────────────────────────────────
        elif cmd == "labels":
            rows = db.labels()
            print(f"\n  Label nel DB ({len(rows)} totali):")
            for r in rows:
                stato = "✅ attivo" if r["is_active"] else "❌ rimosso"
                print(f"    {r['color']:8s} {r['label']:20s} {stato}")
            print()

        else:
            print("  ❓ Comando non riconosciuto. Comandi: dove / storia / spostato / quanti / lista / tutto / labels / exit")


if __name__ == "__main__":
    main()
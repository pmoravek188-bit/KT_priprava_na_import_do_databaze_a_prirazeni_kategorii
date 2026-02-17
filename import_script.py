import pandas as pd
import os
import time
import re
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl import load_workbook

try:
    from ai_categorizer import predict_category, predict_categories_batch
except Exception:
    predict_category = None
    predict_categories_batch = None

# Jednoduchá statistika (jen pro přehled v terminálu)
AI_STATS = {
    "ai_ok": 0,
    "ai_empty": 0,
    "ai_error": 0,
}
AI_STATS_LOCK = threading.Lock()
AI_ERROR_LOGGED = 0
AI_ERROR_LOG_LOCK = threading.Lock()
AI_COST = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "cached_tokens": 0,
    "usd": 0.0,
}
AI_COST_LOCK = threading.Lock()
AI_BUDGET_EXCEEDED = False

# Throttle pro AI volání
_last_ai_call_ts = 0.0
_throttle_lock = threading.Lock()

# Načtení lokálního .env bez externích závislostí
def nacti_env_soubor(cesta: str = ".env"):
    """
    Načte KEY=VALUE řádky z .env do os.environ (pokud klíč ještě není nastaven).
    """
    if not os.path.exists(cesta):
        return

    try:
        with open(cesta, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # když .env nejde načíst, skript má dál fungovat
        return

# Soubory co budu zpracovávat
REFRESH_FILE = "nutrition_data_ReFresh.xlsx"
FIT_FILE = "nutrition_data_FIT.xlsx"
OUTPUT_FILE = "result_output.csv"
CONFIDENCE_FILE = "category_confidence_report.csv"
CATEGORIES_FILE = "categories.xlsx"

# Funkce pro úpravu čísla (z čárky na tečku)
def oprav_cislo(hodnota):
    """Převede číslo s čárkou na normální float."""
    # Pokud je to prázdné, vrátím None
    if hodnota is None or pd.isna(hodnota):
        return None
    
    # Pokud je to už číslo, vrátím ho
    if isinstance(hodnota, (float, int)):
        return float(hodnota)
    
    # Jinak to zkusím převést z textu
    text = str(hodnota).strip()
    text = text.replace(",", ".")  # Čárku nahradím tečkou
    
    if text == "" or text.lower() == "nan":
        return None
    
    try:
        return float(text)
    except:
        # Pokud to nejde převést, vrátím None
        return None

def nacti_kategorie(cesta: str):
    """Načte číselník kategorií (id, název, popis)."""
    try:
        df = pd.read_excel(cesta)
    except Exception:
        return []

    cats = []
    for _, r in df.iterrows():
        try:
            cid = int(r.get("id kategorie"))
        except Exception:
            continue
        cats.append(
            {
                "id_kategorie": cid,
                "nazev": str(r.get("název") or ""),
                "popis": str(r.get("popis") or ""),
            }
        )
    return cats


def _log_ai_error_sample(err):
    global AI_ERROR_LOGGED
    with AI_ERROR_LOG_LOCK:
        if AI_ERROR_LOGGED < 3:
            AI_ERROR_LOGGED += 1
            print(f"AI error (ukázka): {err}", flush=True)


def vyber_kategorii(nazev: str, kategorie_ciselnik, allowed_ids, ai_cfg):
    """
    Vybere kategorii:
    - pokud je zapnuto: AI přes OpenAI API (AI_CATEGORIZATION=1)
    - bez prefiltru a bez ručních heuristik: vždy celý číselník
    """
    ai_enabled = ai_cfg["enabled"]
    api_key = ai_cfg["api_key"]
    model = ai_cfg["model"]
    ai_throttle = ai_cfg["throttle"]
    max_attempts = int(ai_cfg.get("max_attempts") or 1)
    max_cost_usd = float(ai_cfg.get("max_cost_usd") or 0)
    price_input = float(ai_cfg.get("price_input") or 0)
    price_output = float(ai_cfg.get("price_output") or 0)
    price_cached_input = float(ai_cfg.get("price_cached_input") or 0)
    prompt_cache_key = ai_cfg.get("prompt_cache_key") or None
    prompt_cache_retention = ai_cfg.get("prompt_cache_retention") or None

    # pokud není AI zapnutá / klíč / modul / číselník, vrať prázdné
    if not ai_enabled or not api_key or predict_category is None or not kategorie_ciselnik:
        return "", None
    if AI_BUDGET_EXCEEDED:
        return "", None

    # AI je primární: povolené kategorie bereme z categories.xlsx (uzavřený seznam)
    last_conf = None
    last_error = None
    for attempt in range(max_attempts):
        try:
            if ai_throttle > 0:
                global _last_ai_call_ts
                with _throttle_lock:
                    now = time.monotonic()
                    wait = ai_throttle - (now - _last_ai_call_ts)
                    if wait > 0:
                        time.sleep(wait)
                    _last_ai_call_ts = time.monotonic()

            cid, conf, usage = predict_category(
                product_name=str(nazev),
                categories=[
                    {
                        "id_kategorie": c["id_kategorie"],
                        "nazev": c["nazev"],
                        "popis": c.get("popis", ""),
                    }
                    for c in kategorie_ciselnik
                ],
                api_key=api_key,
                model=model,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            last_conf = conf
            last_error = None
            _apply_usage(usage, price_input, price_output, price_cached_input, max_cost_usd)
            if AI_BUDGET_EXCEEDED:
                return "", last_conf
            if cid is not None and cid in allowed_ids:
                with AI_STATS_LOCK:
                    AI_STATS["ai_ok"] += 1
                return cid, conf
        except Exception as e:
            last_error = e
        if attempt < max_attempts - 1:
            time.sleep(min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0.0, 0.3))

    if last_error is not None:
        with AI_STATS_LOCK:
            AI_STATS["ai_error"] += 1
        _log_ai_error_sample(last_error)
    else:
        with AI_STATS_LOCK:
            AI_STATS["ai_empty"] += 1
    # AI nesmí shodit celý import
    return "", last_conf

# Vrátí jen viditelné listy v Excelu
def viditelne_listy(cesta):
    wb = load_workbook(cesta, read_only=True, data_only=True)
    listy = [ws.title for ws in wb.worksheets if ws.sheet_state == "visible"]
    wb.close()
    return listy

# Tady budou všechny výsledky
vysledky = []
confidence_rows = []
category_cache = {}

# GLOBÁLNÍ slovník pro všechny produkty ze všech listů
# Klíč = název produktu, hodnota = součty nebo hodnoty z řádku "Celkové výživové hodnoty"
globalni_produkty = {}

print("Refresh data")

START_TS = time.monotonic()

# načti .env (pokud existuje)
nacti_env_soubor(".env")

# AI konfigurace (čtení z env)
AI_CONFIG = {
    "enabled": str(os.getenv("AI_CATEGORIZATION", "")).lower() in {"1", "true", "yes"},
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": os.getenv("OPENAI_MODEL", "GPT-5 mini"),
    "throttle": float(os.getenv("AI_THROTTLE_SECONDS", "0")),
    "concurrency": int(os.getenv("AI_CONCURRENCY", "8")),
    "max_attempts": int(os.getenv("AI_MAX_ATTEMPTS", "2")),
    "batch_size": int(os.getenv("AI_BATCH_SIZE", "10")),
    "prompt_cache_key": os.getenv("AI_PROMPT_CACHE_KEY", ""),
    "prompt_cache_retention": os.getenv("AI_PROMPT_CACHE_RETENTION", "in_memory"),
    "max_cost_usd": float(os.getenv("AI_MAX_COST_USD", "20")),
    "price_input": float(os.getenv("OPENAI_PRICE_INPUT_PER_MILLION", "0.15")),
    "price_output": float(os.getenv("OPENAI_PRICE_OUTPUT_PER_MILLION", "0.60")),
    "price_cached_input": float(os.getenv("OPENAI_PRICE_CACHED_INPUT_PER_MILLION", "0.075")),
    "fallback_individual": str(os.getenv("AI_FALLBACK_INDIVIDUAL", "1")).lower() in {"1", "true", "yes"},
}

# ===== ZPRACOVÁNÍ REFRESH SOUBORŮ =====
excel_refresh = pd.ExcelFile(REFRESH_FILE)
visible_sheets = viditelne_listy(REFRESH_FILE)
print("  Viditelné listy:", ", ".join(visible_sheets))

# Načtu číselník kategorií
kategorie_ciselnik = nacti_kategorie(CATEGORIES_FILE)
allowed_ids = {c["id_kategorie"] for c in (kategorie_ciselnik or []) if c.get("id_kategorie") is not None}
category_name_by_id = {
    c["id_kategorie"]: c.get("nazev", "")
    for c in (kategorie_ciselnik or [])
    if c.get("id_kategorie") is not None
}


def _nacti_confidence_cache(path: str):
    if not os.path.exists(path):
        return
    try:
        df_cache = pd.read_csv(path, sep=";")
    except Exception:
        return
    for _, r in df_cache.iterrows():
        name = str(r.get("název") or "").strip()
        if not name:
            continue
        id_val = r.get("id kategorie")
        if id_val is None or pd.isna(id_val) or str(id_val).strip() == "":
            continue
        try:
            cid = int(float(id_val))
        except Exception:
            continue
        conf_val = r.get("confidence")
        conf = None
        if conf_val is not None and not pd.isna(conf_val) and str(conf_val).strip() != "":
            try:
                conf = float(conf_val)
            except Exception:
                conf = None
        category_cache[name] = (cid, conf)

_nacti_confidence_cache(CONFIDENCE_FILE)


def _apply_usage(usage, price_input, price_output, price_cached_input, max_cost_usd):
    if not usage:
        return
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    cached_tokens = int(usage.get("cached_tokens") or 0)
    billable_input = max(0, input_tokens - cached_tokens)
    cost = (
        (billable_input / 1_000_000.0) * price_input
        + (cached_tokens / 1_000_000.0) * price_cached_input
        + (output_tokens / 1_000_000.0) * price_output
    )

    global AI_BUDGET_EXCEEDED
    with AI_COST_LOCK:
        AI_COST["input_tokens"] += input_tokens
        AI_COST["output_tokens"] += output_tokens
        AI_COST["total_tokens"] += total_tokens
        AI_COST["cached_tokens"] += cached_tokens
        AI_COST["usd"] += cost
        if max_cost_usd > 0 and AI_COST["usd"] >= max_cost_usd:
            AI_BUDGET_EXCEEDED = True


def _precompute_categories(names):
    names_to_fetch = [n for n in names if n not in category_cache]
    if not names_to_fetch:
        return
    concurrency = max(1, int(AI_CONFIG.get("concurrency") or 1))
    batch_size = max(1, int(AI_CONFIG.get("batch_size") or 1))
    price_input = float(AI_CONFIG.get("price_input") or 0)
    price_output = float(AI_CONFIG.get("price_output") or 0)
    price_cached_input = float(AI_CONFIG.get("price_cached_input") or 0)
    max_cost_usd = float(AI_CONFIG.get("max_cost_usd") or 0)
    prompt_cache_key = AI_CONFIG.get("prompt_cache_key") or "kt_categories_v1"
    prompt_cache_retention = AI_CONFIG.get("prompt_cache_retention") or None
    fallback_individual = bool(AI_CONFIG.get("fallback_individual"))
    total = len(names_to_fetch)

    def _progress_update(done, total_count, last_pct):
        if total_count <= 0:
            return last_pct
        pct = int(done * 100 / total_count)
        if pct >= last_pct + 5 or done == total_count:
            print(f"  AI progress: {done}/{total_count} ({pct}%)", flush=True)
            return pct
        return last_pct

    batches = [names_to_fetch[i:i + batch_size] for i in range(0, len(names_to_fetch), batch_size)]

    def _run_batch(batch):
        if AI_BUDGET_EXCEEDED:
            return {}, None
        if predict_categories_batch is None:
            raise RuntimeError("Batch predikce není k dispozici")
        result_map, usage = predict_categories_batch(
            product_names=batch,
            categories=kategorie_ciselnik,
            api_key=AI_CONFIG["api_key"],
            model=AI_CONFIG["model"],
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )
        _apply_usage(usage, price_input, price_output, price_cached_input, max_cost_usd)
        return result_map, usage

    done = 0
    last_pct = -5

    if concurrency == 1 or len(batches) == 1:
        for batch in batches:
            try:
                result_map, _ = _run_batch(batch)
                for name, (cid, conf) in result_map.items():
                    category_cache[name] = (cid if cid else "", conf)
                    with AI_STATS_LOCK:
                        if cid:
                            AI_STATS["ai_ok"] += 1
                        else:
                            AI_STATS["ai_empty"] += 1
            except Exception as e:
                with AI_STATS_LOCK:
                    AI_STATS["ai_error"] += 1
                _log_ai_error_sample(e)
                for name in batch:
                    category_cache[name] = ("", None)
            done += len(batch)
            last_pct = _progress_update(done, total, last_pct)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_map = {ex.submit(_run_batch, b): b for b in batches}
            for fut in as_completed(future_map):
                batch = future_map[fut]
                try:
                    result_map, _ = fut.result()
                    for name, (cid, conf) in result_map.items():
                        category_cache[name] = (cid if cid else "", conf)
                        with AI_STATS_LOCK:
                            if cid:
                                AI_STATS["ai_ok"] += 1
                            else:
                                AI_STATS["ai_empty"] += 1
                except Exception as e:
                    with AI_STATS_LOCK:
                        AI_STATS["ai_error"] += 1
                    _log_ai_error_sample(e)
                    for name in batch:
                        category_cache[name] = ("", None)
                done += len(batch)
                last_pct = _progress_update(done, total, last_pct)

    # fallback pro chybějící (AI-only)
    if fallback_individual and not AI_BUDGET_EXCEEDED:
        missing = [n for n in names_to_fetch if not category_cache.get(n) or not category_cache[n][0]]
        if missing:
            print(f"  AI fallback (individuálně): {len(missing)} položek", flush=True)
        for n in missing:
            category_cache[n] = vyber_kategorii(n, kategorie_ciselnik, allowed_ids, AI_CONFIG)

# Projdu všechny listy v souboru
for nazev_listu in excel_refresh.sheet_names:
    if nazev_listu not in visible_sheets:
        continue
    # Některé listy nechci zpracovávat
    if nazev_listu in ["Suroviny", "TESTOVÁNÍ", "PV - přehled"]:
        continue
    
    print(f"  Zpracovávám list: {nazev_listu}")
    
    # Načtu data z listu bez hlavičky
    data = pd.read_excel(excel_refresh, sheet_name=nazev_listu, header=None)
    
    # Proměnné pro práci s produkty
    aktualni_produkt = None
    
    # Projdu všechny řádky
    for i in range(len(data)):
        radek = data.iloc[i]
        
        # Ve sloupci 0 je název produktu
        if pd.notna(radek.iloc[0]):
            text = str(radek.iloc[0]).strip().lower()
            # Přeskočím hlavičku
            if text == "název produktu":
                continue
            
            nazev_noveho_produktu = str(radek.iloc[0]).strip()
            
            # Pokud tento produkt už jsem zpracoval, přeskočím ho celý
            if nazev_noveho_produktu in globalni_produkty:
                aktualni_produkt = None
                continue
            
            aktualni_produkt = nazev_noveho_produktu
        
        # Ve sloupci 1 je název suroviny / celkové hodnoty
        if aktualni_produkt and pd.notna(radek.iloc[1]):
            surovina_nazev = str(radek.iloc[1]).strip()
            
            # Pokud ještě nemám tento produkt v GLOBÁLNÍM seznamu, vytvořím ho
            if aktualni_produkt not in globalni_produkty:
                globalni_produkty[aktualni_produkt] = {
                    "porce": None,
                    "kj": None,
                    "tuky": None,
                    "sacharidy": None,
                    "bilkoviny": None,
                    "sul": None,
                    "cukr": None,
                    "nasycene": None
                }

            # Pokud je to řádek s celkovými hodnotami, použiju ho
            if "Celkové výživové hodnoty" in surovina_nazev:
                globalni_produkty[aktualni_produkt]["porce"] = oprav_cislo(radek.iloc[13])
                globalni_produkty[aktualni_produkt]["kj"] = oprav_cislo(radek.iloc[15])
                globalni_produkty[aktualni_produkt]["tuky"] = oprav_cislo(radek.iloc[17])
                globalni_produkty[aktualni_produkt]["nasycene"] = oprav_cislo(radek.iloc[18])
                globalni_produkty[aktualni_produkt]["sacharidy"] = oprav_cislo(radek.iloc[19])
                globalni_produkty[aktualni_produkt]["cukr"] = oprav_cislo(radek.iloc[20])
                globalni_produkty[aktualni_produkt]["bilkoviny"] = oprav_cislo(radek.iloc[21])
                globalni_produkty[aktualni_produkt]["sul"] = oprav_cislo(radek.iloc[22])

# Teď přidám VŠECHNY produkty do výsledků (už bez duplikátů)
produkt_nazvy = list(globalni_produkty.keys())
_precompute_categories(produkt_nazvy)

for nazev_produktu in produkt_nazvy:
    hodnoty = globalni_produkty[nazev_produktu]
    id_kat, conf = category_cache.get(nazev_produktu, ("", None))
    vysledky.append({
        "název": nazev_produktu,
        "porce": hodnoty["porce"],
        "kj": hodnoty["kj"],
        "bílkoviny": hodnoty["bilkoviny"],
        "sacharidy": hodnoty["sacharidy"],
        "tuky": hodnoty["tuky"],
        "provozovny": "ReFresh Bistro",
        "vláknina": "",
        "cukr": hodnoty["cukr"],
        "vápník": "",
        "sůl": hodnoty["sul"],
        "nasycené mastné kyseliny": hodnoty["nasycene"],
        "trans mastné kyseliny": "",
        "mono nasycené": "",
        "poly nasycené": "",
        "cholesterol": "",
        "sodík": "",
        "URL obrázku": "",
        "id kategorie": id_kat,
    })
    kat_nazev = category_name_by_id.get(id_kat, "") if id_kat else ""
    confidence_rows.append(
        {
            "název": nazev_produktu,
            "id kategorie": id_kat,
            "název kategorie": kat_nazev,
            "confidence": conf if conf is not None else "",
            "zdroj": "ReFresh Bistro",
        }
    )

print(f"ReFresh hotový, zpracováno {len(vysledky)} produktů.")

# ===== ZPRACOVÁNÍ FIT SOUBORU =====
print("\nZpracovávám fit data")

fit_data = pd.read_excel(FIT_FILE, header=None)

# FIT má strukturu: řádek 1 = název, řádek 2 = hodnoty, řádek 3 = hodnoty na 100g
i = 0
pocet_fit = 0
fit_names = []

while i < len(fit_data):
    nazev = fit_data.iloc[i, 1]
    if pd.notna(nazev):
        fit_names.append(str(nazev).strip())
    i += 3

_precompute_categories([n for n in fit_names if n])

i = 0

while i < len(fit_data):
    # Přečtu název produktu
    nazev = fit_data.iloc[i, 1]  # Sloupec 1
    
    # Pokud není název, jdu dál
    if pd.isna(nazev):
        i += 1
        continue
    
    nazev = str(nazev).strip()
    
    # Zkusím najít velikost porce v názvu (např. "100 g")
    hledani = re.search(r"(\d+)\s*g", nazev, re.IGNORECASE)
    porce = int(hledani.group(1)) if hledani else None
    
    # Další řádek má hodnoty
    if i + 1 < len(fit_data):
        radek_hodnot = fit_data.iloc[i + 1]
        
        # Přečtu kJ
        kj = oprav_cislo(radek_hodnot.iloc[3])
        
        # Pokud nemám porci, zkusím ji vypočítat z kJ
        if not porce and i + 2 < len(fit_data):
            kj_na_100g = oprav_cislo(fit_data.iloc[i + 2, 3])
            if kj and kj_na_100g and kj_na_100g > 0:
                porce = round(100 * kj / kj_na_100g)
        
    # Přidám do výsledků
    if nazev not in category_cache:
        category_cache[nazev] = vyber_kategorii(nazev, kategorie_ciselnik, allowed_ids, AI_CONFIG)
    id_kat, conf = category_cache.get(nazev, ("", None))
    vysledky.append({
            "název": nazev,
            "porce": porce or "",
            "kj": kj or "",
            "bílkoviny": oprav_cislo(radek_hodnot.iloc[10]) or "",
            "sacharidy": oprav_cislo(radek_hodnot.iloc[7]) or "",
            "tuky": oprav_cislo(radek_hodnot.iloc[5]) or "",
            "provozovny": "",
            "vláknina": oprav_cislo(radek_hodnot.iloc[9]) or "",
            "cukr": oprav_cislo(radek_hodnot.iloc[8]) or "",
            "vápník": "",
            "sůl": oprav_cislo(radek_hodnot.iloc[11]) or "",
            "nasycené mastné kyseliny": oprav_cislo(radek_hodnot.iloc[6]) or "",
            "trans mastné kyseliny": "",
            "mono nasycené": "",
            "poly nasycené": "",
            "cholesterol": "",
            "sodík": "",
            "URL obrázku": "",
            "id kategorie": id_kat,
    })
    kat_nazev = category_name_by_id.get(id_kat, "") if id_kat else ""
    confidence_rows.append(
        {
            "název": nazev,
            "id kategorie": id_kat,
            "název kategorie": kat_nazev,
            "confidence": conf if conf is not None else "",
            "zdroj": "FIT",
        }
    )
    pocet_fit += 1
    
    # Přeskočím na další produkt (po 3 řádcích)
    i += 3

print(f"FIT hotové, zpracováno {pocet_fit} produktů.")

# Statistiky AI kategorizace
print(
    "\nAI kategorizace (statistika): "
    f"ai_ok={AI_STATS['ai_ok']}, ai_empty={AI_STATS['ai_empty']}, "
    f"ai_error={AI_STATS['ai_error']}"
)
print(
    "AI usage: "
    f"input={AI_COST['input_tokens']}, cached={AI_COST['cached_tokens']}, "
    f"output={AI_COST['output_tokens']}, "
    f"total={AI_COST['total_tokens']}, cost=${AI_COST['usd']:.4f}"
)
if AI_BUDGET_EXCEEDED:
    print("Pozor: dosažen limit AI_MAX_COST_USD, další volání byla zastavena.")
elapsed_s = time.monotonic() - START_TS
print(f"Čas běhu: {elapsed_s:.1f}s")

# ===== VYTVOŘENÍ VÝSTUPU =====
print(f"\nCelkem produktů: {len(vysledky)}")
print("Vytvoření csv souboru")

# Vytvořím DataFrame
df = pd.DataFrame(vysledky)

# Přidám ID na začátek
df.insert(0, "id", range(1, len(df) + 1))

# Seřadím sloupce podle vzoru
sloupce = [
    "id", "název", "porce", "kj", "bílkoviny", "sacharidy", "tuky", "provozovny",
    "vláknina", "cukr", "vápník", "sůl", "nasycené mastné kyseliny",
    "trans mastné kyseliny", "mono nasycené", "poly nasycené",
    "cholesterol", "sodík", "URL obrázku", "id kategorie",
]
df = df[sloupce]

# Uložím do CSV
df.to_csv(OUTPUT_FILE, sep=";", index=False)

print(f"Hotovo {OUTPUT_FILE}")

# Uložím report s confidence (mimo importní CSV)
if confidence_rows:
    df_conf = pd.DataFrame(confidence_rows)
    df_conf.to_csv(CONFIDENCE_FILE, sep=";", index=False)
    print(f"Hotovo {CONFIDENCE_FILE}")

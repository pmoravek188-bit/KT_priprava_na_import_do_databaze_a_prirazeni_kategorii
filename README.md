# Příprava dat na import do databáze a přiřazení kategorií

Systém pro zpracování nutričních dat z Excel souborů a automatickou kategorizaci produktů pomocí AI.

## Co projekt umí

- **Import nutričních dat** z Excel souborů (`nutrition_data_ReFresh.xlsx` a `nutrition_data_FIT.xlsx`)
- **Automatická kategorizace produktů** pomocí OpenAI API
- **Výpočet nutričních hodnot** na porci
- **Výstup do CSV** formátu připraveného pro import do databáze
- **Report s confidence skóre** pro každou kategorizaci

## Hlavní funkce

### Zpracování dat
- Načítá produkty z více listů v Excel souborech
- Zpracovává pouze viditelné listy (ignoruje skryté)
- Sčítá nutriční hodnoty ze surovin pro každý produkt
- Podporuje různé formáty vstupních dat (ReFresh vs FIT)

### AI kategorizace
- Používá OpenAI API pro inteligentní přiřazení kategorií
- **Doporučený model:** `o1-mini` (pro nejlepší výsledky)
- Podporuje batch zpracování pro rychlejší kategorizaci
- Ukládá confidence skóre pro každou kategorizaci
- Má ochranu proti překročení rozpočtu (AI_MAX_COST_USD)
- Podporuje prompt cache pro úsporu nákladů

## Jak spustit

### Požadavky
```bash
pip install pandas openpyxl
```

### Konfigurace
Vytvořte soubor `.env` v této složce s následujícími proměnnými (volitelné):
```
AI_CATEGORIZATION=1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=o1-mini  # Doporučeno pro nejlepší výsledky
AI_MAX_COST_USD=20
AI_BATCH_SIZE=10
AI_CONCURRENCY=8
```

### Spuštění
```bash
python import_script.py
```

## Vstupní soubory

- `nutrition_data_ReFresh.xlsx` - produkty z ReFresh Bistro
- `nutrition_data_FIT.xlsx` - produkty z FIT
- `categories.xlsx` - číselník kategorií (id, název, popis)

## Výstupní soubory

- `result_output.csv` - hlavní výstup s produkty a nutričními hodnotami
- `category_confidence_report.csv` - report s kategorizacemi a confidence skóre

## Struktura projektu

- `import_script.py` - hlavní skript pro zpracování dat
- `ai_categorizer.py` - modul pro AI kategorizaci pomocí OpenAI API

## Dokumentace

Více informací najdete v souboru:
- `dokumentace_aktualizovana.pdf` - detailní dokumentace projektu

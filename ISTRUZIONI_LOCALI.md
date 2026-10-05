# Istruzioni operative (file locale, non versionato)

Guida passo passo per far girare lo studio sulla macchina Linux con RTX 3090: installazione,
pilota, esperimenti, annotazioni dei medici e analisi finale.
Il protocollo scientifico è in `METHODOLOGY.md`; qui c'è solo il "come si fa".

---

## 0. Tenere questo file fuori da Git

Metti il file nella cartella principale della repo e aggiungilo all'esclusione locale, che vale
solo sul tuo computer e non modifica il `.gitignore` condiviso:

```bash
echo "ISTRUZIONI_LOCALI.md" >> .git/info/exclude
git status        # il file non deve comparire
```

Su Windows (PowerShell):

```powershell
Add-Content .git\info\exclude "ISTRUZIONI_LOCALI.md"
```

---

## 1. Ordine delle operazioni

| # | Cosa | Dove | Quanto dura |
|---|---|---|---|
| 1 | Preparare la macchina Linux | Linux | 30 min |
| 2 | Server vLLM con Qwen | Linux (GPU) | 15 min + download |
| 3 | Ambiente `prompteval` + test | Linux | 10 min |
| 4 | Pilota (20 domande × 2 run) | Linux | 30–60 min |
| 5 | Esperimento principale | Linux | ore (Qwen) + ore (Claude, in parallelo) |
| 6 | Urgenza di riferimento (medici) | Excel | si può fare subito, in parallelo |
| 7 | Scelta condizione migliore → triage → braccio thinking | Linux | ore |
| 8 | Danno dei distrattori (medici) | Excel | dopo i run |
| 9 | Analisi finale | Linux | minuti |
| 10 | Backup dei risultati | ovunque | sempre |

---

## 2. Preparare la macchina Linux

Controlla GPU e driver:

```bash
nvidia-smi
```

Devi vedere la RTX 3090 con 24 GB e una versione CUDA. Se il comando non esiste, mancano i driver
NVIDIA: vanno installati prima di tutto (chiedi all'amministratore della macchina se non è tua).

Strumenti di base:

```bash
sudo apt update && sudo apt install -y git tmux curl
```

Miniconda, se non c'è già:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh      # accetta e chiudi/riapri il terminale
```

Clona la repo (è privata: serve un token GitHub o una chiave SSH):

```bash
cd ~
git clone https://github.com/LM-Healthcare/LLM-EVAL-PROMPT.git
cd LLM-EVAL-PROMPT
echo "ISTRUZIONI_LOCALI.md" >> .git/info/exclude
```

---

## 3. Server vLLM con Qwen3.5-9B (ambiente separato)

vLLM fissa le sue versioni di torch/CUDA, quindi ha un ambiente tutto suo.

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
```

Se all'avvio vLLM non riconosce il modello Qwen3.5, installa la nightly come indicato nel model card:

```bash
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
```

### Avvio del server (esperimento principale e triage)

Sempre dentro `tmux`, così non si ferma se chiudi la connessione:

```bash
tmux new -s qwen
conda activate vllm
vllm serve Qwen/Qwen3.5-9B --port 8000 --max-model-len 8192 \
  --language-model-only --reasoning-parser qwen3 --gpu-memory-utilization 0.92
```

- Il primo avvio scarica il modello (~18 GB) in `~/.cache/huggingface`.
- `--language-model-only` disattiva il modulo visivo e libera memoria (le immagini sono escluse).
- Per uscire da tmux lasciando il server acceso: `Ctrl+B`, poi `D`. Per rientrare: `tmux attach -t qwen`.

### Verifica che il server risponda

Da un altro terminale:

```bash
curl http://localhost:8000/v1/models
```

Deve comparire `Qwen/Qwen3.5-9B`. Prova una domanda con il ragionamento spento:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3.5-9B",
  "messages": [{"role": "user", "content": "Rispondi solo con: ANSWER: A"}],
  "max_tokens": 50, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

La risposta non deve contenere blocchi di ragionamento (`<think>`).

---

## 4. Ambiente `prompteval`, test e dataset

```bash
conda create -n prompteval python=3.12 -y
conda activate prompteval
cd ~/LLM-EVAL-PROMPT
pip install -e ".[all,dev]"
pytest                                        # atteso: 23 passed
prompteval fetch-dataset --dest data/ITAMed   # scarica ITAMed (pubblico)
```

### Chiave API di Anthropic

Il codice legge la chiave dalla variabile d'ambiente `ANTHROPIC_API_KEY` (non da un file `.env`).
Per averla sempre disponibile:

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
echo $ANTHROPIC_API_KEY | cut -c1-10           # controllo: mostra solo l'inizio
```

Non scrivere mai la chiave in file della repo.

### Split di valutazione

`data/splits/eval_split.json` è già nella repo: 300 domande di test, 864 + 96 per il fine-tuning.
**Non rigenerarlo.** Se cambia la versione di ITAMed (vedi §11), va deciso prima di partire con
l'esperimento principale, non dopo.

---

## 5. Pilota

Serve a controllare parametri, formato delle risposte, troncamenti, tempi e costi.
Usa le prime 20 domande dello split, 2 run, tutte le condizioni, entrambe le lingue
(320 chiamate per modello). I risultati vanno in `results/pilot` e **non** fanno parte dello studio.

```bash
conda activate prompteval
cd ~/LLM-EVAL-PROMPT

prompteval dry-run -c configs/pilot.yaml
# guarda i prompt generati in results/pilot/dry_run/ (un file per modello, lingua e condizione)

prompteval run -c configs/pilot.yaml --model qwen35_9b
prompteval run -c configs/pilot.yaml --model sonnet55
prompteval analyze -c configs/pilot.yaml
```

### Cosa controllare dopo il pilota

1. **Errori bloccanti.** Il riepilogo di `run` deve dire `"ok": 320` per ogni modello. Se compare
   `fatal_error`, c'è un parametro rifiutato dall'API: copia il messaggio e sistemiamo la config.
2. **Chiamate fallite.** `results/pilot/errors.jsonl` non deve esistere (o deve essere vuoto).
3. **Formato.** In `results/pilot/analysis/report.md` la colonna "Parse fail %" dovrebbe essere
   sotto il 2%.
4. **Troncamenti.** In `results/pilot/analysis/usage.csv` la colonna `truncated` deve essere 0.
   Se non lo è (tipicamente in C1 o C3), alza `max_tokens` nella config.
5. **Lettura a campione.** Leggi qualche risposta a occhio:

```bash
python - <<'EOF'
import json, random
rows = [json.loads(l) for l in open("results/pilot/responses/qwen35_9b.jsonl")]
for r in random.sample(rows, 3):
    print("=" * 80)
    print(r["condition"], r["language"], r["question_code"], "corretta:", r["correct_letter"])
    print(r["response_text"][:1500])
EOF
```

6. **Tempi e costi.** In `usage.csv` trovi i token per condizione: moltiplica per 24.000/320 = 75
   per stimare l'esperimento principale per modello.

Se il pilota va bene, la cartella `results/pilot` si può lasciare lì (è ignorata da Git).

---

## 6. Esperimento principale

48.000 risposte (24.000 per modello). I due modelli girano in parallelo, ognuno nella sua
sessione tmux:

```bash
tmux new -s run-qwen
conda activate prompteval && cd ~/LLM-EVAL-PROMPT
prompteval run -c configs/main.yaml --model qwen35_9b
```

```bash
tmux new -s run-claude
conda activate prompteval && cd ~/LLM-EVAL-PROMPT
prompteval run -c configs/main.yaml --model sonnet55
```

Controllo dell'avanzamento, da qualsiasi terminale:

```bash
prompteval status -c configs/main.yaml
```

**Se si interrompe** (crash, riavvio, rete): rilancia esattamente lo stesso comando. Le risposte
già salvate non vengono rifatte. Se ci sono righe in `results/main/errors.jsonl`, rilanciare il
comando ritenta anche quelle.

**Non modificare** `prompts/prompts.yaml`, lo split o la config del modello durante
l'esperimento: ogni risposta registra l'hash del prompt, e un cambio a metà rende i dati non
confrontabili.

Quando entrambi hanno finito:

```bash
prompteval analyze -c configs/main.yaml
prompteval select-best -c configs/main.yaml     # scrive results/main/best_conditions.json
```

Un'analisi con 2.000 repliche bootstrap può richiedere diversi minuti.

---

## 7. Urgenza di riferimento (medici) — si può fare subito

Genera le schede (una per medico):

```bash
prompteval export-urgency -c configs/main.yaml --raters R1 R2
# -> annotations/urgency/urgency_R1.xlsx, urgency_R2.xlsx
```

- Ogni medico compila la sua scheda in autonomia, senza confrontarsi con l'altro. La prima scheda
  del file Excel contiene le istruzioni.
- Valori ammessi: `emergency`, `urgent`, `non-urgent`, `not-applicable` (menu a tendina).
- Puoi rinominare i medici (`--raters Rossi Bianchi`): il nome finisce nel nome del file.

Quando tornano compilate, rimettile nella stessa cartella e importa:

```bash
prompteval import-urgency --files annotations/urgency/urgency_R1.xlsx annotations/urgency/urgency_R2.xlsx
```

Il comando stampa l'accordo (κ di Cohen) e, se ci sono disaccordi, crea
`annotations/urgency/urgency_adjudication.xlsx`. Il terzo medico compila la colonna
`final_decision`, poi:

```bash
prompteval import-urgency --files annotations/urgency/urgency_R1.xlsx annotations/urgency/urgency_R2.xlsx \
  --adjudication annotations/urgency/urgency_adjudication.xlsx
```

Risultato: `annotations/urgency_final.csv`, letto automaticamente dall'analisi del triage.

---

## 8. Esperimenti secondari

Entrambi usano `results/main/best_conditions.json`, quindi vanno lanciati **dopo** `select-best`.

### Triage

Il server vLLM resta quello del §3:

```bash
prompteval run -c configs/triage.yaml --model qwen35_9b
prompteval run -c configs/triage.yaml --model sonnet55
```

### Braccio con ragionamento attivo

Qwen può scrivere fino a 16.000 token: **riavvia vLLM** con un contesto più lungo.

```bash
tmux attach -t qwen        # Ctrl+C per fermare il server, poi:
vllm serve Qwen/Qwen3.5-9B --port 8000 --max-model-len 20480 \
  --language-model-only --reasoning-parser qwen3 --gpu-memory-utilization 0.92
```

```bash
prompteval run -c configs/thinking_arm.yaml --model qwen35_9b_thinking
prompteval run -c configs/thinking_arm.yaml --model sonnet55_thinking
```

Se compaiono errori di memoria (OOM) o timeout, abbassa `concurrency` del modello in
`configs/thinking_arm.yaml` (da 16 a 4–8) e rilancia: riparte da dove era arrivato.

---

## 9. Danno dei distrattori (medici) — dopo tutti i run

Genera le schede con tutti i distrattori scelti almeno una volta, da tutti gli esperimenti:

```bash
prompteval export-distractors -c configs/main.yaml \
  --from results/main results/triage results/thinking_arm --raters R1 R2
# -> annotations/distractors/distractor_harm_R1.xlsx, distractor_harm_R2.xlsx
```

- I medici vedono solo caso, risposta corretta e risposta sbagliata: non sanno quale modello o
  condizione l'ha scelta.
- Punteggio 0 / 1 / 2 secondo la scala nella prima scheda del file.
- Il numero di righe può essere di diverse centinaia: conviene dividere il lavoro in più sessioni.

Import e terzo medico, come per l'urgenza:

```bash
prompteval import-distractors --files annotations/distractors/distractor_harm_R1.xlsx \
  annotations/distractors/distractor_harm_R2.xlsx
# se ci sono disaccordi: compilare distractor_harm_adjudication.xlsx, poi
prompteval import-distractors --files annotations/distractors/distractor_harm_R1.xlsx \
  annotations/distractors/distractor_harm_R2.xlsx \
  --adjudication annotations/distractors/distractor_harm_adjudication.xlsx
```

Risultato: `annotations/distractor_harm_final.csv`.

Se dopo l'import lanci altri run (o ne rifai qualcuno), esporta solo i distrattori nuovi:

```bash
prompteval export-distractors -c configs/main.yaml --from results/main results/triage results/thinking_arm --only-new
```

### Riferimenti di C3 da verificare

```bash
prompteval export-references -c configs/main.yaml --per-model 150
# -> annotations/references/references_to_verify.csv
```

Per ogni riferimento compilare `verdict` con `exists`, `exists_wrong_details` o `not_found`.

---

## 10. Analisi finale

```bash
prompteval analyze -c configs/main.yaml
prompteval analyze -c configs/triage.yaml
prompteval analyze -c configs/thinking_arm.yaml
```

Per ogni esperimento, in `results/<nome>/analysis/`:
- `report.md`: tabelle riassuntive (accuratezza, stabilità, calibrazione, danno, contrasti);
- `metrics_by_cell.csv`: tutte le metriche con intervalli di confidenza al 95%;
- `contrasts.csv`: confronti con C0, italiano vs inglese, coppie di modelli;
- `gee_condition_effects.csv`, `gee_interactions.txt`: modelli statistici;
- grafici `reliability_*.png` e `accuracy_by_condition.png`.

### Analisi di sensibilità in R (facoltativa)

```bash
sudo apt install -y r-base
Rscript -e 'install.packages(c("lme4", "emmeans", "car"), repos = "https://cloud.r-project.org")'
Rscript analysis_r/glmm.R results/main/analysis/responses_long.csv C0
```

Questo script non è stato testato automaticamente: se dà errore, mandami l'output.

---

## 11. Prima di partire con l'esperimento principale

- [ ] **Versione di ITAMed.** La config fissa il commit `4d70e66…`. Se la versione definitiva su
  Zenodo corrisponde a un altro commit, aggiorna `dataset.commit` in `configs/main.yaml` ed esegui
  `git -C data/ITAMed checkout <commit>`. Se il contenuto delle domande è cambiato, va rigenerato
  lo split (`prompteval make-split ... --force`) **prima** di qualsiasi run vero.
- [ ] Pilota completato e controllato (§5).
- [ ] `max_tokens` adeguato (nessun troncamento nel pilota).
- [ ] Medici reclutati e informati; rubrica del danno provata su ~30 distrattori.
- [ ] (Consigliato) registrazione del protocollo su OSF prima del run principale.
- [ ] Date di esecuzione annotate: vanno nel paper.

---

## 12. Backup dei risultati

La cartella `results/` è esclusa da Git: **se si perde la macchina, si perdono le risposte.**
Fai una copia alla fine di ogni esperimento (e durante, se dura giorni):

```bash
tar czf ~/backup_results_$(date +%Y%m%d).tar.gz results annotations
# poi copiala altrove, ad esempio sul tuo PC:
scp utente@macchina-linux:~/backup_results_*.tar.gz C:\Users\filow\Desktop\
```

Anche `annotations/` va salvata: contiene le schede compilate dai medici (gli `.xlsx` non sono
versionati), mentre i CSV finali sì.

---

## 13. Problemi frequenti

| Sintomo | Causa probabile | Cosa fare |
|---|---|---|
| `Connection refused` su localhost:8000 | server vLLM spento o ancora in caricamento | `tmux attach -t qwen`, aspetta il messaggio di server pronto |
| `CUDA out of memory` in vLLM | contesto o batch troppo grandi | abbassa `--gpu-memory-utilization` a 0.88, riduci `--max-model-len` o `concurrency` |
| `fatal_error: HTTP 400 ...` per Claude | parametro non accettato dall'API | mandami il messaggio: si corregge `params`/`extra_body` nella config |
| `Missing API key` | variabile non impostata in quella sessione | `source ~/.bashrc` o `export ANTHROPIC_API_KEY=...` |
| Righe in `errors.jsonl` (429, timeout) | limiti di velocità o rete | rilancia lo stesso comando; se ricorrono, abbassa `concurrency` o `requests_per_minute` |
| Parse failure alto per un modello | non rispetta il blocco finale `ANSWER`/`CONFIDENCE` | guarda alcune risposte (§5) e mandamele |
| `FileExistsError` su `make-split` | lo split esiste già (protezione voluta) | non rigenerarlo; `--force` solo prima dei run veri |
| Avvisi di deepeval durante `pytest` | plugin di un altro pacchetto nell'ambiente | usa l'ambiente `prompteval` dedicato |
| `ITAMed not found` | dataset non scaricato | `prompteval fetch-dataset --dest data/ITAMed` |

---

## 14. Comandi in sintesi

```bash
# ogni volta che apri un terminale
conda activate prompteval && cd ~/LLM-EVAL-PROMPT

prompteval dry-run  -c configs/<config>.yaml        # nessuna chiamata, mostra conteggi e prompt
prompteval run      -c configs/<config>.yaml [--model <id>] [--max-tasks N]
prompteval status   -c configs/<config>.yaml
prompteval analyze  -c configs/<config>.yaml
prompteval select-best -c configs/main.yaml
prompteval export-urgency / import-urgency
prompteval export-distractors / import-distractors
prompteval export-references
```

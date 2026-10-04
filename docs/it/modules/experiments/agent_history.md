# Performance Modello

> Appunti storici: esempi, run e parametri si riferiscono agli esperimenti
> descritti, non alle istruzioni di avvio attuali.

> Versione anonimizzata: nomi, identificativi, area geografica, codici gerarchici e codici prodotto sono stati sostituiti con etichette fittizie coerenti. Settori, categorie e risultati quantitativi sono stati mantenuti per preservare il valore dell’analisi.


1. Valutazione richieste basate su count e primi movimenti sul grafo
Per testare il comportamento del modello su richieste basate su conteggi, utilizziamo la seguente domanda:

**Domanda**: Qual è il miglior venditore assegnato all’Area geografica A nel settore Metallo?
**Risposta Agente**:
```md
**Risultato:** il miglior venditore dell’Area geografica A nel settore Metallo è **VENDITORE A**, ID **VEND-001**.
**Evidenze:** gestisce **137 clienti distinti**; il risultato è il primo per numero di clienti tra i venditori collegati all’Area geografica A e al settore Metallo.
**Prossimo passo:** usalo come riferimento per confrontare portafoglio e opportunità sul territorio. Dati alla mano, è il benchmark da analizzare.
```

Note
La domanda non rappresenta un caso d’uso reale, ma serve come test interno per verificare se il modello è in grado di identificare correttamente un soggetto di riferimento per l’interazione.
In assenza di metriche dirette sulle performance, consideriamo come miglior venditore quello con il maggior numero di clienti nel settore Metallo nell’Area geografica A.


2. Impersonifichiamo il venditore e chiediamoli una panoramica informativa
**Domanda**: Ciao sono Venditore A fammi una panoramica sulla base clienti, sul mio settore operativo e sulla mia scala gerarchica

**Risposta Agente**:
```md
**Risultato** → Ciao Venditore A: sei il venditore **ID VEND-001**, operativo nell’**Area geografica A** e nel settore **Metallo**, con una base di **137 clienti** e un forte spazio di sviluppo su fissaggio, utensili e lavorazione metallo.

**Evidenze**
- **Base clienti:** 137 clienti assegnati.
- **Mix settoriale:**
  - Metallo: **88 clienti**
  - Installazioni: 16
  - Legno: 14
  - Costruzioni: 13
  - Cargo: 6
- **Segmenti principali:**
  - Settore Metallo – ambito costruzione: **46 clienti**
  - Manutenzione: 26
  - Settore Metallo – ambito produzione: 16
  - Imprese edili/Grandi cantieri: 10
- **Canali:** 59 clienti HW su canale 3, 52 HW su canale 2, 23 HW su canale 4 e 3 PO.
- **Filiali più rappresentate:**
  - Costruzioni finestre metalliche, codice **CAT-001**: 22 clienti
  - Costruzioni in metallo/acciaio, codice **CAT-002**: 17
  - Costruzioni apparecchi, codice **CAT-003**: 10
  - Produzione alimentare e bevande, codice **CAT-004**: 8
- **Categorie Level5 più diffuse nella base:**
  - Silicone neutro serramento, codice **PROD-001**: 45 clienti
  - Dado DIN 934 in acciaio zincato, codice **PROD-002**: 36
  - Punta a percussione Plus Quadro-L, codice **PROD-003**: 32
  - ISO 4017 acciaio 8.8 zincato, codice **PROD-004**: 31
  - DIN 522 acciaio zincato, codice **PROD-005**: 31
  - Inserto PH C 6.3, codice **PROD-006**: 29
  - Flessometro con carcassa rivestita in gomma, codice **PROD-007**: 29
- **Clienti con maggiore ampiezza d’acquisto:**
  - **Cliente 001**, ID **CLI-001**: 262 categorie
  - **Cliente 002**, ID **CLI-002**: 241
  - **Cliente 003**, ID **CLI-003**: 198
  - **Cliente 004**, ID **CLI-004**: 170
  - **Cliente 005**, ID **CLI-005**: 153
- **Scala gerarchica disponibile:**
  - Città/area: **Area geografica A**
  - Settore: **Metallo**
  - Distretto: **Distretto D-01**
  - Job: **JOB-001**
  - Territorio e macroregione: non risultano valorizzati per il tuo profilo.

**Prossimo passo** → Parti dai clienti del comparto metallo che acquistano già fissaggi o utensili, ma non entrambe le famiglie: costruisci una proposta combinata su viteria, inserti, punte e dischi da taglio. Dai subito priorità a **Cliente 001**, **Cliente 002** e **Cliente 003**: sono i clienti con la maggiore profondità di acquisto.
```

3) Introduciamo le Raccomandazioni

**Domanda** Oggi devo lavorare nel settore Metallo, dimmi quali sono i 5 prodotti più raccomandato che mi permettono di coprire più clienti di quel settore in un solo giro

**Risposta Agente**:
```md
**Risultato** → Per coprire più clienti del settore **Metallo** in un solo giro, parti da questi 5 prodotti raccomandati:

1. **DIN 522 acciaio zincato** — codice **PROD-005**: **64 clienti**
2. **Cutter con manico in plastica e cursore** — codice **PROD-008**: **60 clienti**
3. **DIN 934 acciaio I6I / I8I zincato** — codice **PROD-002**: **58 clienti**
4. **ISO 4017 acciaio 8.8 zincato** — codice **PROD-004**: **50 clienti**
5. **Punta elicoidale HSS DIN338 RN 130°** — codice **PROD-009**: **50 clienti**

**Evidenze**
- I conteggi si riferiscono a clienti distinti del settore **Metallo** assegnati al Venditore A.
- Sono prodotti collegati a raccomandazioni presenti nel database.
- Il dato misura la copertura potenziale per prodotto; i clienti possono comparire in più categorie, quindi i conteggi non sono sommabili.

**Prossimo passo** → Organizza il giro partendo da **DIN 522**, **Cutter** e **DIN 934**: sono le tre categorie con la maggiore copertura. Porta poi in chiusura **ISO 4017** e **punta HSS** come proposta complementare. Hai una copertura ampia: vai sui clienti con una proposta combinata.
```

## 4

**Domanda** A quali clienti posso raccomandare tutti e 5 questi prodotti?

**Risposta Agente**:
```md
**Risultato** → Hai **46 clienti del settore Metallo** collegati a tutti e 5 i prodotti, tramite acquisto e/o raccomandazione. Per il giro di oggi, dai priorità ai clienti che ne hanno già acquistati 1–4: lì hai il cross-selling più concreto.

**Evidenze**

**Priorità alta – hanno acquistato 1 prodotto**
- **Cliente 006** — ID **CLI-006**
- **Cliente 007** — ID **CLI-007**
- **Cliente 008** — ID **CLI-008**
- **Cliente 009** — ID **CLI-009**
- **Cliente 010** — ID **CLI-010**
- **Cliente 011** — ID **CLI-011**
- **Cliente 012** — ID **CLI-012**
- **Cliente 013** — ID **CLI-013**

**Priorità alta – hanno acquistato 2 prodotti**
- **Cliente 014** — ID **CLI-014**
- **Cliente 015** — ID **CLI-015**
- **Cliente 010** — ID **CLI-016**
- **Cliente 017** — ID **CLI-017**
- **Cliente 018** — ID **CLI-018**

**Priorità media – hanno acquistato 3 prodotti**
- **Cliente 005** — ID **CLI-005**
- **Cliente 019** — ID **CLI-019**
- **Cliente 020** — ID **CLI-020**
- **Cliente 021** — ID **CLI-021**
- **Cliente 022** — ID **CLI-022**
- **Cliente 023** — ID **CLI-023**

**Priorità media – hanno acquistato 4 prodotti**
- **Cliente 024** — ID **CLI-024**
- **Cliente 025** — ID **CLI-025**
- **Cliente 026** — ID **CLI-026**
- **Cliente 026** — ID **CLI-026**
- **Cliente 021** — ID **CLI-021**

**Clienti con tutti e 5 già acquistati**
- **Cliente 003** — ID **CLI-003**
- **Cliente 004** — ID **CLI-004**
- **Cliente 027** — ID **CLI-027**
- **Cliente 028** — ID **CLI-028**
- **Cliente 029** — ID **CLI-029**

**Clienti collegati ai 5 prodotti ma senza acquisti registrati sui cinque**
- **Cliente 030** — ID **CLI-030**
- **Cliente 031** — ID **CLI-031**
- **Cliente 032** — ID **CLI-032**
- **Cliente 033** — ID **CLI-033**
- **Cliente 034** — ID **CLI-034**
- **Cliente 035** — ID **CLI-035**
- **Cliente 036** — ID **CLI-036**
- **Cliente 037** — ID **CLI-037**
- **Cliente 038** — ID **CLI-038**
- **Cliente 039** — ID **CLI-039**
- **Cliente 040** — ID **CLI-040**
- **Cliente 041** — ID **CLI-041**
- **Cliente 042** — ID **CLI-042**
- **Cliente 043** — ID **CLI-043**
- **Cliente 044** — ID **CLI-044**
- **Cliente 045** — ID **CLI-045**
- **Cliente 046** — ID **CLI-046**
- **Cliente 047** — ID **CLI-047**
- **Cliente 048** — ID **CLI-048**

Nota: **Cliente 026** e **Cliente 021** risultano duplicati nella lista di origine; considerali una sola volta.

**Prossimo passo** → Costruisci il giro su **Cliente 006, Cliente 007, Cliente 008, Cliente 011, Cliente 012, Cliente 013, Cliente 005 e Cliente 021**: hanno già una base d’acquisto e un margine evidente per completare la gamma. Porta i cinque prodotti e proponi il pacchetto completo.
```

5)

**Domanda** Analizziamo il Cliente 031: quali sono le sue categorie più acquistate?

**Risposta Agente**:
```md
**Risultato** → Per **Cliente 031**, ID **CLI-031**, non emerge una categoria più acquistata delle altre: le categorie registrate hanno tutte **1 acquisto/connessione** nel database.

**Evidenze**
Tra le categorie acquistate risultano:
- **Archetto semiauricolare x-300** — codice **PROD-010**
- **Armadio da officina Basic** — codice **PROD-011**
- **Assortimento chiavi a bussola 1/4", 33 pezzi** — codice **PROD-012**
- **Cacciavite a taglio lama tonda** — codice **PROD-013**
- **Chiave combinata corta** — codice **PROD-014**
- **Fascetta cablaggio KBL1 PA linguetta in plastica** — codice **PROD-015**
- **Nastro sigillafiletti in PTFE** — codice **PROD-016**
- **Pinza universale ZEBRA DIN ISO 5746** — codice **PROD-017**
- **Serie chiavi combinate corte, 11 pezzi** — codice **PROD-018**
- **Spazzola a mano universale in acciaio** — codice **PROD-019**

Le date di acquisto registrate arrivano fino al **maggio 2026**, ad esempio per **Archetto semiauricolare x-300** e **Tamponi di ricambio per archetto x-300**.

**Prossimo passo** → Non impostare la visita sulla categoria “più acquistata”: punta sul consolidamento della fornitura, partendo da utensili manuali, DPI, fascette e prodotti per pulizia/manutenzione. Il cliente ha già acquistato in molte famiglie: c’è spazio per una proposta più strutturata.
```

6) Analisi cross sell e reasoning sui prodotti

**Domanda**: ottimo ora fai un’analisi cross-sell sui 5 prodotti raccomandati sempre per Cliente 031: individua i prodotti correlati, ad esempio per categoria, e restituisci le migliori strategie di vendita da proporre al cliente.

**Risposta Agente**:
```md
**Risultato** → Cliente 031 ha già acquistato prodotti vicini a **3 dei 5 target**: il cross-sell più concreto è su **utensili da taglio**, **lavorazione materiali** e completamento della **viteria**. Tutti e 5 i prodotti risultano raccomandati, ma **nessuno è ancora acquistato**.

**Evidenze**

| Prodotto raccomandato | Stato | Categoria | Evidenza utile |
|---|---|---|---|
| **DIN 522 acciaio zincato** — **PROD-005** | Raccomandato, non acquistato | Level2 **Rondelle / rosette** — Level1 **Elementi di fissaggio** | Non risultano altri acquisti nella stessa categoria Level2 |
| **Cutter con manico in plastica e cursore** — **PROD-008** | Raccomandato, non acquistato | Level2 **Utensili da taglio** — Level1 **Utensili a mano** | Ha già acquistato 2 prodotti della categoria: **Cutter coltello a serramanico tascabile combinato** — **PROD-020**, ultimo acquisto ottobre 2025; **Coltello sicurezza Martor Secunorm Smartcut** — **PROD-021**, ultimo acquisto novembre 2025 |
| **DIN 934 acciaio I6I/I8I zincato** — **PROD-002** | Raccomandato, non acquistato | Level2 **Dadi** — Level1 **Elementi di fissaggio** | Non risultano acquisti nella stessa categoria Level2 |
| **ISO 4017 acciaio 8.8 zincato** — **PROD-004** | Raccomandato, non acquistato | Level2 **Viti metriche/pollice** — Level1 **Elementi di fissaggio** | Ha già acquistato **DIN 965 acciaio 4.8 zincato H** — **PROD-022**, ultimo acquisto marzo 2026 |
| **Punta elicoidale HSS DIN338 RN 130°** — **PROD-009** | Raccomandata, non acquistata | Level2 **Punte per forare e scalpellare** — Level1 **Lavorazione materiali** | Ha già acquistato **Assortimento punte percussione DUO-S cilindrico, 10 pezzi Syska** — **PROD-023**, ultimo acquisto novembre 2025 |

**Strategie di vendita**

1. **Parti dal cutter: è il cross-sell più caldo.**
   Il cliente ha già acquistato due prodotti per il taglio. Proponi il **Cutter con manico in plastica e cursore** come alternativa più strutturata o come prodotto standard per le attività ricorrenti.
   **Ipotesi commerciale:** lavorare su sostituzione, disponibilità immediata e uniformazione dell’attrezzatura.

2. **Apri la proposta sulla foratura professionale.**
   Ha già acquistato un assortimento di punte percussione; proponi la **Punta HSS DIN338** per coprire una diversa esigenza di foratura.
   **Ipotesi commerciale:** posizionarla come complemento per lavorazioni su metallo, senza presentarla come sostituto delle punte già acquistate.

3. **Costruisci un kit di fissaggio completo.**
   Il cliente ha già acquistato una **DIN 965**, quindi puoi proporre **ISO 4017**, **DIN 934** e **DIN 522** come ampliamento della dotazione: vite, dado e rondella.
   **Ipotesi commerciale:** la vendita a kit è più efficace della proposta dei tre articoli separati, anche se il database non dimostra un consumo specifico di dadi e rondelle.

4. **Usa la viteria come proposta di completamento, non come priorità iniziale.**
   **ISO 4017** ha il legame più diretto con l’acquisto già registrato della DIN 965. Presenta poi **DIN 934** e **DIN 522** per chiudere il sistema di fissaggio.

**Prossimo passo** → In visita proponi prima **Cutter + Punta HSS**, sostenendoti sugli acquisti già registrati. Poi chiudi con un **kit ISO 4017 + DIN 934 + DIN 522**. Parti da ciò che il cliente compra già: hai evidenze concrete per entrare nella trattativa.
```

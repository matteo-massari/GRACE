# Storico del modello

**Storia del modello**
Parametri da tenere fissi:
- n_batch: 1024
- n_hop: 2 
- n_sample_at_node: 30
- lr: 0.0001
- eta_min: 0.000001
- warmup_period: 3
- n_epochs: 50
- n_clienti: 1000 / None


Confronto tra lr cosine e lr fisso
- Stack: embedding -> L2 Normalizzazione -> conv -> ReLU -> L2 normalizzazione -> linear
- Mlflow run:
    - 2026-07-21@08:46_RecommenderModel_ncl:1000_euclidean_lr:0.0001 (non cosine)
    - 2026-07-20@14:08_RecommenderModel_ncl:1000_feat_lr:alto_eucledian (cosine)


Fine tuning del margin della Triplet loss:
- Reason: miglior margine da turilizzare nella triplet loss
- Stack: Embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear -> L2 normalizzazione
- ML flow run:
    - 2026-05-26@13:20_LinkModelContrastive_ncl:1000_tripletLoss_m:0.2
    - 2026-05-26@11:42_LinkModelContrastive_ncl:1000_tripletLoss_m:0.4
    - 2026-05-26@11:41_LinkModelContrastive_ncl:1000_tripletLoss_m:0.6
- Conclusione: Anche se impercettibile vediamo che un margine a 0.4 produce degli score più alti


L2 normalization vs LayerNorm vs BatchNorm
- Reason: Capire quale normalizzazione utilizzare nel layer di convoluzione
- Stack: embedding -> SageConv -> (Layer da controllate) -> linear -> L2 normalizzazione
- Loss: Triplet loss
- Ml flow run:
    - 2026-05-26@09:54_LinkModelContrastive_ncl:_tripletLoss_BatchNorm_noPr
    - 026-05-26@09:18_LinkModelContrastive_ncl:1000_tripletLoss_LayerNorm_noPr
    - 2026-05-26@08:53_LinkModelContrastive_ncl:1000_tripletLoss_l2norm_noPr
- Conlusione: L2 regolarizzazione nella convoluzione porta a risultati migliori


Final normalization vs no normalization
- Reason: Capire se conviene o no tenere l'ultima normalizzazione in training
- Stack: embedding -> SageConv -> l2 normalization -> linear -> (layer da contorllare)
- Loss: Triplet loss
- Ml flow run:
    - 2026-06-29@10:44_RecommenderModel_ncl:1000_tripeltloss_norm_inference
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
- Conlusione: Se utilizzata la l2 normalization porta a risultati peggiori. 


Triplet Loss vs InfoNCE vs bayesian personalized loss vs Binary cross entropy 
- reason: Verificare se funziona meglio la InfoNCE o la TripletLoss
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear
    No two tower e no feature
- Ml flow run:
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
    - 2026-05-22@16:37_LinkModelContrastive_ncl:1000_InfoNCE_l2norm_noPr (unica normalizzata per correttezza)
    - 2026-06-29@09:24_RecommenderModel_ncl:1000_bprLoss_NOl2norm_noPr
    - 2026-06-30@13:27_RecommenderModel_ncl:1000_bce_NOl2norm_noPr
- Conlusione: La **triplet loss** produce dei risultati migliori, quindi teniamo questa come Loss


Aggregazione SageConv
- Reasons: verificare qual'è la migliore aggregazione in per SageConv tra max, mean o sum.
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear
- Ml flow run:
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr (mean)
    - 2026-07-07@08:37_RecommenderModel_ncl:1000_aggr:max
    - lstm va out of memory
- Conclusion: con max crollo delle prformance, quindi preferiamo mean


Projection Layer
- Reason: Allungando la rete, otteniamo un punteggio più alto?
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear -> (layerNorm -> ReLU -> linear) x pr
- Loss: Triplet loss
- ML flow run:
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
    - 2026-06-30@13:35_RecommenderModel_ncl:1000_tripletLoss_pr:1
    - 2026-06-30@14:17_RecommenderModel_ncl:1000_tripletLoss_pr:2
- Conclusion: Aggiungendo layer di proizione perdiamo in metriche. Questo potrebbe essere collegato ad un Overfitting (quindi dobbiamo avere una rete più corta ed usare i layer lineari solamente per avere una riduzione dellop spazio vettoriale) oppure introduiciamo rumore e quindi conviene provare un approccio a two tower


Architettura ibrida two tower or tying strategies
- Reason: potrebbe essere necessario avere dei layer separati per gli oggetti e gli utenti
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear (diviso per item e per user)
- Loss: Triplet Loss
- ML flow run:
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
    - 2026-06-30@14:39_RecommenderModel_ncl:1000_tripletLoss_two_tower
- Conclusion: Qua non ci sta una evidenza piena, quindi bisogna vedere come si comporta con un'allungamento della rete


Tow tower diverse normalizzazioni
- Reason: Testiamo che tipo di normalizzazione usare nell'allungamento della reta quando si utilizza il twotower
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear -> layerNorm -> ReLU -> linear -> (None/batchNorm/layerNorm)
- Runs:
    - 2026-06-02@20:14_LinkModelContrastive_ncl:1000_two_tower_no_feat_pr:1_None_final
    - 2026-06-02@19:29_LinkModelContrastive_ncl:1000_two_tower_no_feat_pr:1_batch_final
    - 2026-06-02@18:20_LinkModelContrastive_ncl:1000_two_tower_no_feat_pr:1_layer_final
- Conclusion: in questo caso la normalizzazione rappresenta il problema in quanto senza normalizzazione il modello ha delle performance migliore.


Allungamento della rete con l'architettura two tower
- Reason: il two tower potrebbe avere prestazioni migliori con un allungamento della testa di proiezione finale. Questo ci permetterebbe anche di valutare la precedente esplorazione sull'allungamento della rete, andando a verificare se effettivamente è un problema di overfitting o semplicemente mischiavamo informazioni
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear -> (layerNorm -> ReLU -> linear) x pr
- Runs
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr 
    - 2026-06-30@14:39_RecommenderModel_ncl:1000_tripletLoss_two_tower
    - 2026-06-30@16:25_RecommenderModel_ncl:1000_tripletLoss_two_tower_pr:1
    - 2026-06-30@16:26_RecommenderModel_ncl:1000_tripletLoss_two_tower_pr:2
- Conclusion: Le conclusioni riportano al fatto che allungando la proiezioni le prestazioni non migliorano, anzi confermano che produce solo bias e lo porta in overfitting.

Conclusione sulla two tower: Non si evidenzia un grande miglioramento, al fronte di una complicazione della rete, quindi come opzione la teniamo aperta ma non la implementiamo per ora. 

Verifica grandezza degli embedding
- Reason: Verifichiamo come migliora il training andando ad aumentare la grandezza degli embedding
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear
- Ml Runs:
    - 2026-07-06@14:43_RecommenderModel_ncl:1000_1024_512_256
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr (512-256-128)
    - 2026-07-06@15:58_RecommenderModel_ncl:1000_256_128_64
- Conclusione: Lo spazio miigliore equivale a 512-256-128 in quanto è il migliore in performance

Utilizzo della cosine o dell'euclidean distance
- Reason: capire che distanza usare in inferenza
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear
- Ml runs:
    - 2026-07-14@09:04_RecommenderModel_ncl:1000_no_feat_dist:cosine
    - 2026-07-14@09:03_RecommenderModel_ncl:1000_no_feat_dist:euclidean
- Conclusion: 
    è migliore l'euclidean 

Da qua la migliore architettura che si vuole mandare in produzione è *2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr*

-- Inserimento delle feature -- 
Le feature sono state implementate nel miglior modo possibile ed è sono state applicate quelle corrette. Infatti fanno riferimento anche al file di inserimento delle feature, l'analisi statistica ha mostrato come le feature che sembrano avere una maggire correlazione (ha parità di multicollinearià) sono:
- **customer_att_sector**: settore di appartenenza
- **customer_att_potential**: quantità di persone che utilizzano i prodotti wurth all'interno dell'azienda * 600
- **customer_att_county**: Città
- **customer_att_creditlimit**: correlato alla quantità spesa


Da qua l'aggiunta delle feature ha avuto della ricerca strutturale quindi non sono state utilizzate quelle del notebook
Aggiunta di feature dai dati degli utenti e degli oggetti:
- Reason: Vedere se l'aggiunta di dati avrebbe comportato un miglioamento dei suggerimenti
- Stack: embedding -> conv -> ReLU -> L2 normalizzazione -> linear 
- Loss: Triplet Loss
- ML flow run:
    - 2026-05-39@11:50_LinkModelContrastive_ncl:1000_with_feat
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
Conclusion: Analizzando le metriche di prestazione, alcune sono sovrapposte, altre sono leggermente più basse. Il suo utilizzo va bene, perchè le prestazioni non calano, ma necessitano un tuning degli iperparametri. Adesso come hp abbiamo, le feature stesse, le feature stocastiche, e in questo caso la grandezza dei canali riceve un'importanza superiore. Inoltre si vede necessario un lavoro di feature engegneering e di selezione delle variabili, ma l'integrazione di quest'utlime potrebbe essere utile


Da qua le feature sono state implementate nel miglior modo possibile ed è sono state applicate quelle corrette. Infatti fanno riferimento anche al file di inserimento delle feature, l'analisi statistica ha mostrato come le feature che sembrano avere una maggire correlazione (ha parità di multicollinearià) sono:
- **customer_att_sector**: settore di appartenenza
- **customer_att_potential**: quantità di persone che utilizzano i prodotti wurth all'interno dell'azienda * 600
- **customer_att_county**: Città
- **customer_att_creditlimit**: correlato alla quantità spesa

Implementazioni feature
- Reason: Introduzione delle feature scelte dalle statistiche invece di quelle a caso
- Stack: embedding_feat -> conv -> ReLU -> L2 normalizzazione -> linear
- ML flow run:
    - 2026-06-19@09:06_RecommenderModel_ncl:1000_top_feat
    - 2026-06-08@15:30_RecommenderModel_ncl:1000_TOP
- Conclusion: I risultati sembrano essere più bassi ma facendo riferimento ai notebbok 


Architettura ibrida two tower
- Reason: potrebbe essere necessario avere dei layer separati per gli oggetti e gli utenti, quindi confrontiamo un'architettura two tower senza feat e con le feat
- Stack: embedding_feat -> conv -> ReLU -> L2 normalizzazione -> linear (diviso)
- Ml flow run:
    - 2026-06-10@10:51_RecommenderModel_ncl:1000_with_feat_tt
    - 2026-07-06@13:53_RecommenderModel_ncl:1000_with_feat
    OPPURE
    - 2026-07-01@11:15_RecommenderModel_ncl:None_two_tower_feat
    - 2026-06-12@08:46_RecommenderModel_ncl:None_feat_nc:2
- Conclusione: Vediamo nell'architettura two tower un potenziale sviluppo in quanto gli score di precisione e di recall in quanto in entrambi i casi sono più alti, però dagli altri esperimenti vediamo con un'allungamneto della rete porta ad un overfitting. Quindi semplicemente cambia l'ultimo layer di classificazione finale. 


Aumento numero di connessioni 
- Reason: la sparsità della rete può risultare in una aggiunta di rumore e quindi una perdita di informazione, quindi (senza architettura two tower) andiamo a fare delle valutazioni per capire la quantita corretta di connessioni minime per clienti utilizzate
- Stack: embedding_feat -> conv -> ReLU -> L2 normalizzazione -> linear
- Ml flow run:
    - 2026-07-06@09:24_RecommenderModel_ncl:None_feat_nc:5
    - 2026-07-04@12:46_RecommenderModel_ncl:None_feat_nc:4
    - 2026-06-12@13:59_RecommenderModel_ncl:None_feat_nc:3
    - 2026-06-12@08:46_RecommenderModel_ncl:None_feat_nc:2
- Conclusione:
    Come possiamo vedere c'è un miglioramento man mano che le connessioni aumentano, di conseguenza è come se il numero di connessioni aiutasse a caratterizzare megliio il soggetto, e che quindi gli acquisti sono più correlati. Ovviamente c'è un peggioramento per quanto riguarda i prodotti più popolari. Questa però risulta molto una scelta si business, quindi manteniamo i nc = 2. Ma deve essere preso in considerazione quando si sceglie un modello migliore. 

Normalizzazione delle feature
- Reason: Partendo dal modello base, utilizziamo le feature e capiamo se devono essere normalizzate o è meglio lasciarle libere
- Stack: embedding -> (L2 Normalizzazione) -> conv -> ReLU -> L2 normalizzazione -> linear 
- Loss Triplet Loss
- Ml flow runs:
    - 2026-07-20@14:08_RecommenderModel_ncl:1000_feat_lr:alto_eucledian (feature Normalizzate)
    - 2026-07-06@13:53_RecommenderModel_ncl:1000_with_feat (feature non normalizzate)
- Conclusione:
    Utilizziamo le feature normalizzate e facciamo un controllo con un ncl più ampio


Correzione learning rate
- Reason: Con un aumento del numero di connessioni il training perde stabilità, quindi abbassiamo il learning rate da 0.0001 - 0.00001 a 0.00001 - 0.000001
- Stack: embedding_feat -> L2 normalizzazione -> conv -> ReLU -> L2 normalizzazione -> linear
- Ml flow run:
    - 2026-07-04@12:46_RecommenderModel_ncl:None_feat_nc:4
    - 2026-07-21@15:52_RecommenderModel_ncl:None_euclidean_lr:var
- Conclusione:
    utilizzare un lr più basso porta sia a dei risultati più stabili e sia a performance migliori, intolre è stata introdotta la normalizzazione delle feature. 

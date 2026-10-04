# Archive: notes of the GNN model

> Historical notes: examples, run and parameters refer to experiments
> described, not to current startup instructions.

**Model history**
Parameters to keep fixed:
- n_batch: 1024
- n_hop: 2
- n_sample_at_node: 30
- lr: 0.0001
- eta_min: 0.000001
- warmup_period: 3
- n_epochs: 50
- n_customers: 1000 / None



Fine tuning of the Triplet loss margin:
- Reason: best margin to be moved into triplet loss
- Stack: Embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear -> L2 normalizzazione
- ML flow run:
    - 2026-05-26@13:20_LinkModelContrastive_ncl:1000_tripletLoss_m:0.2
    - 2026-05-26@11:42_LinkModelContrastive_ncl:1000_tripletLoss_m:0.4
    - 2026-05-26@11:41_LinkModelContrastive_ncl:1000_tripletLoss_m:0.6
- Conclusion: Although imperceptible we see that a margin at 0.4 produces higher scores


L2 normalization vs LayerNorm vs BatchNorm
- Reason: Understand which normalization use in the convolution layer
- Stack: embedding -> SageConv -> (Layer by subsidiaries) -> linear -> L2 normalization
- Loss: Triplet loss
- Ml flow run:
    - 2026-05-26@09:54_LinkModelContrastive_ncl:_tripletLoss_BatchNorm_noPr
    - 026-05-26@09:18_LinkModelContrastive_ncl:1000_tripletLoss_LayerNorm_noPr
    - 2026-05-26@08:53_LinkModelContrastive_ncl:1000_tripletLoss_l2norm_noPr
- Conclusion: L2 regularization in convolution leads to better results


Final normalization vs no normalization
- Reason: Understand whether or not to keep the last normalization in training
- Stack: embedding -> SageConv -> l2 normalization -> linear -> (contorllar layer)
- Loss: Triplet loss
- Ml flow run:
    - 2026-06-29@10:44_RecommenderModel_ncl:1000_tripeltloss_norm_inference
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
- Conclusion: If used l2 normalization leads to worse results.


Triplet Loss vs InfoNCE vs bayesian personalized loss vs Binary cross entropy
- reason: Check if the InfoNCE or TripletLoss works best
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear
    No two tower and no feature
- Ml flow run:
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
    - 2026-05-22@16:37_LinkModelContrastive_ncl:1000_InfoNCE_l2norm noPr (only normalized for correctness)
    - 2026-06-29@09:24_RecommenderModel_ncl:1000_bprLoss_NOl2norm_noPr
    - 2026-06-30@13:27_RecommenderModel_ncl:1000_bce_NOl2norm_noPr
- Conclusion: The **triplet loss** produces better results, so we keep this as Loss


Aggregazione SageConv
- Reasons: verify what is the best aggregation in for SageConv between max, mean or sum.
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear
- Ml flow run:
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr (mean)
    - 2026-07-07@08:37_RecommenderModel_ncl:1000_aggr:max
    - lstm va out of memory
- Conclusion: with max collapse of prformance, so we prefer mean


Projection Layer
- Reason: By extending the network, do we get a higher score?
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear -> (layerNorm -> ReLU -> linear) x pr
- Loss: Triplet loss
- ML flow run:
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
    - 2026-06-30@13:35_RecommenderModel_ncl:1000_tripletLoss_pr:1
    - 2026-06-30@14:17_RecommenderModel_ncl:1000_tripletLoss_pr:2
- Conclusion: Adding proition layer we lose in metrics. This could be connected to an Overfitting (so we have to have a shorter network and use linear layers only to have a reduction in vector space) or introduce noise and therefore it is best to try a two tower approach


Hybrid architecture two tower or tying strategies
- Reason: You may need separate layers for objects and users
- Stack: embedding -> SageConv -> ReLU -> L2 standardization -> linear (divided for item and user)
- Loss: Triplet Loss
- ML flow run:
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
    - 2026-06-30@14:39_RecommenderModel_ncl:1000_tripletLoss_two_tower
- Conclusion: There is no full evidence here, so you have to see how it behaves with an extension of the network


Tow tower different normalizations
- Reason: We test what type of normalization to use in the stretching of the reta when using the twotower
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear -> layerNorm -> ReLU -> linear -> (None/batchNorm/layerNorm)
- Runs:
    - 2026-06-02@20:14_LinkModelContrastive_ncl:1000_two_tower_no_feat_pr:1_None_final
    - 2026-06-02@19:29_LinkModelContrastive_ncl:1000_two_tower_no_feat_pr:1_batch_final
    - 2026-06-02@18:20_LinkModelContrastive_ncl:1000_two_tower_no_feat_pr:1_layer_final
- Conclusion: In this case normalization represents the problem as without normalization the model has better performances.


Network extension with two tower architecture
- Reason: The two tower could have better performance with a final projection head stretching. This would also allow us to evaluate the previous exploration of the elongation of the network, going to check whether it is actually an overfitting problem or simply mixing information
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear -> (layerNorm -> ReLU -> linear) x pr
- Runs
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
    - 2026-06-30@14:39_RecommenderModel_ncl:1000_tripletLoss_two_tower
    - 2026-06-30@16:25_RecommenderModel_ncl:1000_tripletLoss_two_tower_pr:1
    - 2026-06-30@16:26_RecommenderModel_ncl:1000_tripletLoss_two_tower_pr:2
- Conclusion: The conclusions refer to the fact that by extending the projections the performance does not improve, indeed confirm that it produces only bias and leads it in overfitting.

Conclusion on the two tower: There is no great improvement, in the face of a network complication, so as an option we keep it open but we do not implement it for now.

Check embedding size
- Reason: We test how to improve training by increasing the size of embedding
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear
- Ml Runs:
    - 2026-07-06@14:43_RecommenderModel_ncl:1000_1024_512_256
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr (512-256-128)
    - 2026-07-06@15:58_RecommenderModel_ncl:1000_256_128_64
- Conclusion: The mileage space is equivalent to 512-256-128 as it is the best in performance

Use of sodium or Euclidean distance
- Reason: Understand what distance to use in inference
- Stack: embedding -> SageConv -> ReLU -> L2 normalizzazione -> linear
- Ml runs:
    - 2026-07-14@09:04_RecommenderModel_ncl:1000_no_feat_dist:cosine
    - 2026-07-14@09:03_RecommenderModel_ncl:1000_no_feat_dist:Euclidean
- Conclusion:
    is better the Euclidean

From here the best architecture you want to send to production is *2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr*

-- Inserting the features --
The features have been implemented in the best possible way and the correct ones have been applied. In fact they also refer to the feature insertion file, statistical analysis showed how the features that seem to have a maggire correlation (has multi-cholinear parity) are:
- **customer_att_sector**: business sector
- **customer_att_potential**: quantity of people using wurth products within the company * 600
- **customer_att_county**: City
- **customer_att_creditlimit**: related to the amount spent


From here the addition of the features had structural research so those of the notebook were not used
Added features from user data and objects:
- Reason: See if adding data would result in a misinterpretation of suggestions
- Stack: embedding -> conv -> ReLU -> L2 normalizzazione -> linear
- Loss: Triplet Loss
- ML flow run:
    - 2026-05-39@11:50_LinkModelContrastive_ncl:1000_with_feat
    - 2026-06-29@09:25_RecommenderModel_ncl:1000_tripletLoss_NOl2norm_noPr
Conclusion: Analyzing performance metrics, some are superimposed, others are slightly lower. Its use is good, because the performance does not fall, but require a tuning of the hyperparameters. Now as hp we have, the features themselves, the stochastic features, and in this case the size of the channels receives a higher importance. In addition, there is a need for a feature engegneering and variable selection work, but the integration of this autlime could be useful.


From here the features have been implemented in the best possible way and the correct ones have been applied. In fact they also refer to the feature insertion file, statistical analysis showed how the features that seem to have a maggire correlation (has multi-cholinear parity) are:
- **customer_att_sector**: business sector
- **customer_att_potential**: quantity of people using wurth products within the company * 600
- **customer_att_county**: City
- **customer_att_creditlimit**: related to the amount spent

Implementazioni feature
- Reason: Introduction of features chosen by statistics instead of random ones
- Stack: embedding_feat -> conv -> ReLU -> L2 normalizzazione -> linear
- ML flow run:
    - 2026-06-19@09:06_RecommenderModel_ncl:1000_top_feat
    - 2026-06-08@15:30_RecommenderModel_ncl:1000_TOP
- Conclusion: The results seem to be lower but referring to the notebook


Two tower hybrid architecture
- Reason: It may be necessary to have separate layers for objects and users, so we compare a two tower architecture without feat and feats
- Stack: embedding_feat -> conv -> ReLU -> L2 normalizzazione -> linear (diviso)
- Ml flow run:
    - 2026-06-10@10:51_RecommenderModel_ncl:1000_with_feat_tt
    - 2026-07-06@13:53_RecommenderModel_ncl:1000_with_feat
    OPPURE
    - 2026-07-01@11:15_RecommenderModel_ncl:None_two_tower_feat
    - 2026-06-12@08:46_RecommenderModel_ncl:None_feat_nc:2
- Conclusion: We see in the two tower architecture a potential development as precision scores and recalls as in both cases are higher, but from the other experiments we see with an elongated network leads to an overfitting. Then simply change the final classification layer.


Increase number of connections
- Reason: network sparsity can result in an addition of noise and therefore a loss of information, so (without two tower architecture) we go to make evaluations to understand the correct amount of minimum connections for customers used
- Stack: embedding_feat -> conv -> ReLU -> L2 normalizzazione -> linear
- Ml flow run:
    - 2026-07-06@09:24_RecommenderModel_ncl:None_feat_nc:5
    - 2026-07-04@12:46_RecommenderModel_ncl:None_feat_nc:4
    - 2026-06-12@13:59_RecommenderModel_ncl:None_feat_nc:3
    - 2026-06-12@08:46_RecommenderModel_ncl:None_feat_nc:2
- Conclusione:
    As we can see there is an improvement as connections increase, so it is as if the number of connections helped to characterize the subject, and that therefore the purchases are more related. Obviously there is a worsening regarding the most popular products. But this is very a business choice, so we keep nc = 2. But it must be taken into account when choosing a better model.

Normalization of features
- Reason: Starting from the basic model, we use the features and we understand whether they have to be normalized or it is better to leave them free
- Stack: embedding -> (L2 Normalizzazione) -> conv -> ReLU -> L2 normalizzazione -> linear
- Loss Triplet Loss
- Ml flow runs:
    - 2026-07-20@14:08_RecommenderModel_ncl:1000_feat_lr:alto_eucledian (feature Normalizzate)
    - 2026-07-06@13:53_RecommenderModel_ncl:1000_with_feat (non-standard)
- Conclusione:
    We use normalized features and check with a wider ncl


Correzione learning rate
- Reason: With an increase in number of connections the training loses stability, then lower the learning rate from 0.0001 - 0.00001 to 0.00001 - 0.000001
- Stack: embedding_feat -> L2 normalizzazione -> conv -> ReLU -> L2 normalizzazione -> linear
- Ml flow run:
    - 2026-07-04@12:46_RecommenderModel_ncl:None_feat_nc:4
    - 2026-07-21@15:52_RecommenderModel_ncl:None_Euclidean_lr:var
- Conclusione:
    use a lower lr leads both to more stable and better performance results, the normalization of the features was introduced.

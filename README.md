<h2 align="center">
<p> SumGNN: Multi-typed Drug Interaction Prediction via Efficient Knowledge Graph Summarization</h2>

This is the code for our paper ``SumGNN: Multi-typed Drug Interaction Prediction via Efficient Knowledge Graph Summarization'' (published in Bioinformatics'21) [[link]](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btab207/6189090).
 
## Install

```bash
git clone git@github.com:yueyu1030/SumGNN.git
cd SumGNN
pip install -r requirements.txt
```

## Example

```
python train.py 
    -d drugbank         # task
    -e ddi_hop3         # the name for the log for experiments
    --gpu=0             # ID of GPU
    --hop=3             # size of the hops for subgraph
    --batch=256         # batch size for samples
    --emb_dim=32        # size of embedding for GNN layers
    -b=10               # size of basis for relation kernel
```

You can also change the ```d``` to BioSNAP. Please change the ```e``` accordingly. The trained model and the logs are stored in experiments folder. Note that to ensure a fair comparison, we test all models on the same negative triplets. 

## Dataset

We provide the dataset in the [data](data/) folder. 

| Data  | Source | Description
|-------|----------|----------|
| [Drugbank](data/drugbank/) | [This link](https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/data/)| A drug-drug interaction network betweeen 1,709 drugs with 136,351 interactions.| 
| [TWOSIDES](data/BioSNAP/) | [This link](http://snap.stanford.edu/biodata/datasets/10018/10018-ChSe-Decagon.html)| A drug-drug interaction network betweeen 645 drugs with 46221 interactions.|
| Hetionet | [This link](https://github.com/hetio/hetionet) | The knowledge graph containing 33,765  nodes  out  of  11  types  (e.g.,  gene,  disease,  pathway,molecular function and etc.) with 1,690,693 edges from 23 relation types after preprocessing (To ensure **no information leakage**, we remove all the overlapping edges  between  HetioNet  and  the  dataset).

## Knowledge Graph Embedding
We train the knowledge graph embedding based on the framework in [OpenKE](https://github.com/thunlp/OpenKE). 

To obtain the embedding on your own, you need to first feed the triples in `train.txt` (edges in dataset) and `relations_2hop.txt` (edges in KG) as edges into their toolkit and obtain the embeddings for each node. Then, you can incorporate this embedding into our framework by modifying the line 44-45
in `model/dgl/rgcn_model.py`.

## Cite Us

Please kindly cite this paper if you find it useful for your research. Thanks!

```
@article{yu2021sumgnn,
  title={Sumgnn: Multi-typed drug interaction prediction via efficient knowledge graph summarization},
  author={Yu, Yue and Huang, Kexin and Zhang, Chao and Glass, Lucas M and Sun, Jimeng and Xiao, Cao},
  journal={Bioinformatics},
  year={2021}
}
```

## Acknowledgement
The code framework is based on [GraIL](https://github.com/kkteru/grail).
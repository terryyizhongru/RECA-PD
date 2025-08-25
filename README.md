<h1 align="center"><span style="font-weight:normal">RECA-PD: A Robust Explainable Cross-Attention Method for Speech-based Parkinson's Disease Classification</h1>
  
<div align="center">
  
[Terry Yi Zhong](https://terryyizhong.github.io/)
</div>

<div align="center">
  
[📘 Introduction](#intro) |
[🛠️ Preparation](#preparation) |
[🚀 Training and Evaluation](#training_and_evaluation) |
[📖 Citation](#citation) |

📣 _Accepted for [TSD 2025](https://www.kiv.zcu.cz/tsd2025/index.php)!_

</div>

## <a name="intro"></a> 📘 Introduction

<div align="center"> <img src="docs/TSD-V1.2.drawio.png"  width="720"> </div>

**Abstract.** _Parkinson's Disease (PD) affects over 10 million people globally, with speech impairments often preceding motor symptoms by years, making speech a valuable modality for early, non-invasive detection. While recent deep-learning models achieve high accuracy, they typically lack the explainability required for clinical use. To address this, we propose RECA-PD, a novel, robust, and explainable cross-attention architecture that combines interpretable speech features with self-supervised representations. RECA-PD matches state-of-the-art performance in Parkinson’s disease detection while providing explanations that are more consistent and more clinically meaningful. Additionally, we demonstrate that performance degradation in certain speech tasks (e.g., monologue) can be mitigated by segmenting long recordings. Our findings indicate that performance and explainability are not necessarily mutually exclusive. Future work will enhance the usability of explanations for non-experts and explore severity estimation to increase the real-world clinical relevance._ [📜 Arxiv](https://arxiv.org/abs/2507.03594) 📜 TSD 2025 (to be update)

## Acknowledgements

Part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research program NGF AiNed Fellowship Grants, which is financed by the Dutch Research Council (NWO). This work used the Dutch national e-infrastructure with the support of the SURF Cooperative using grant no. EINF-10519.


## <a name="preparation"></a> 🛠️ Preparation



### Environment
- Prepare the **conda environment** to run the experiments:

```
conda create -n RECAPD python=3.10
conda activate RECAPD
pip install -r requirements.txt
```

> **Note:** This repository is primarily built upon and extends the work from [interpreting-ssl-parkinson-speech](https://github.com/david-gimeno/interpreting-ssl-parkinson-speech).


### Preprocessing

Data preprocessing, dataset split, feature extraction scripts. As an example, we provide the scripts aimed to address our GITA corpus experiments:

```
bash scripts/runs/dataset_preparation/gita.sh $DATASET_DIR $METADATA_PATH
bash scripts/runs/feature_extraction/gita.sh
```

, where `$DATASET_DIR` and `$METADATA_PATH` refer to the directory containing all the audio waveform samples and the CSV including the corpus subject metadata, respectively. _Please, note that you have to convert the 1st sheet of the .xlsx provided in the GITA dataset to a .csv file._

We also include the scripts used to generate the split-mono training set described in the paper. To train and test on this set:
	1.	Follow the instructions in splits/gita-splitmono/README.md to generate the new audio segments.
	2.	Update the folder paths in the scripts above to point to your newly created segments so you can extract the corresponding features.

Note that, to keep the overall pipeline straightforward, the split-mono task is not integrated into the scripts of Evaluation Protocol 1 introduced below.


## <a name="training_and_evaluation"></a> 🚀 Training and Evaluation


We evaluate four methods (M1–M4) under two protocols as introduced in the paper.  
Here, we show the scripts for the proposed method (RECA-PD); the other methods follow the same procedure within their respective folders.

Generated splits for both protocols are available under:
- `splits/gita_splits1/`
- `splits/gita_splits2/`


### Training

All training scripts are located in:  
- `scripts/runs/experiments/training/protocol1/`  
- `scripts/runs/experiments/training/protocol2/`  

To train each method, run the corresponding shell script:

| Method                         | Script                               |
|--------------------------------|--------------------------------------|
| **M1 (Base)**                  | `gita_base_M1.sh`                    |
| **M2 (M1 + Fixed Softmax)**         | `gita_fix_softmax_M2.sh`             |
| **M3 (M2 + InterpretableValue)**   | `gita_fix_interpretableValue_M3.sh`  |
| **M4 (RECA-PD, proposed)**     | `gita_RECAPD_M4.sh`                  |


Example:
```
bash scripts/runs/experiments/training/protocol2/gita_RECAPD_M4.sh
```

### Evaluation

In order to **evaluate** follow two protocols , you can run the following commands:

```
bash scripts/runs/experiments/eval/print_res_table1.sh exps/gita_splits1/M4/
bash scripts/runs/experiments/eval/print_res_table2.sh exps/gita_splits2/M4/combined_set/
```


### Analysis and Visualization

Wilcoxon signed-rank test example:

- `scripts/evaluation/WS-rank.ipynb`

A simple Visualization example:
- `scripts/interpretability/visual_example.ipynb`


## <a name="citation"></a> 📖 Citation

If you use this dataset or methods from this project in academic work, please cite:
 
### 📄 LaTeX (BibTeX)
```bibtex
@misc{zhong2025recapdrobustexplainablecrossattention,
      title={RECA-PD: A Robust Explainable Cross-Attention Method for Speech-based Parkinson's Disease Classification}, 
      author={Terry Yi Zhong and Cristian Tejedor-Garcia and Martha Larson and Bastiaan R. Bloem},
      year={2025},
      eprint={2507.03594},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2507.03594}, 
}
```

### 📚 APA
Zhong, T. Y., Tejedor-Garcia, C., Larson, M., & Bloem, B. R. (2025). RECA-PD: A robust explainable cross-attention method for speech-based Parkinson’s disease classification. arXiv. https://arxiv.org/abs/2507.03594
Accepted at TSD 2025.


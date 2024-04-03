# ShortGPT
Unofficial implementations of:
- ["ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"](https://arxiv.org/pdf/2403.03853)
- ["The Unreasonable Ineffectiveness of the Deeper Layers"](https://arxiv.org/abs/2403.17887)

### To Use
- Follow Llama 2 setup found [here](https://github.com/facebookresearch/llama).
- Reference `short_gpt/short_llama.ipynb` for necessary function calls.


### Details
- Use a wrapper around Llama to collect hidden states and compute BI (block influence).
  - BI implementation may be subject to change or improvements if others find issues, thanks in advance!
- Sum importance values across layers while inferencing on [pg19](https://huggingface.co/datasets/pg19).
  - Dataset can be slow to load from huggingface so you may want to use an alternative.
- Use sorted layer-wise importance values to determine which layers are least important and subject to removal.
- Demonstrate *model-healing* with Mistral-7B-v0.1 described in "The Unreasonable Ineffectiveness of the Deeper Layers", where finetuning with LoRA after layer removal can recover downstream model performance. 


### Results
Comparison of ShortGPT layers removed on Llama-2-7B (9 least important layers):

Paper: [27, 26, 25, 28, 24, 29, 23, 21, 22] \
This Implementation: [25, 27, 24, 26, 28, 29, 23, 22, 21]

Same layers but different order.

### TODO:
- [x] Is order significant -> Authors mention that layer order varies between datasets but their relative ordering suggests "similar levels of importance" [link](https://huggingface.co/papers/2403.03853#65f028667c916f24c80e93b3).
- [x] Add more models and metrics -> Add experimental support for HF models on this [branch](https://github.com/sramshetty/ShortGPT/tree/hf-models).
  - [x] Add angular distance metric
  - [x] Demonstrate model healing using HuggingFace model [here](https://github.com/sramshetty/ShortGPT/blob/hf-models/short_gpt/short_hf.ipynb).   

### Citations
```bibtex
@misc{men2024shortgpt,
    title={ShortGPT: Layers in Large Language Models are More Redundant Than You Expect}, 
    author={Xin Men and Mingyu Xu and Qingyu Zhang and Bingning Wang and Hongyu Lin and Yaojie Lu and Xianpei Han and Weipeng Chen},
    year={2024},
    eprint={2403.03853},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@misc{gromov2024unreasonable,
    title={The Unreasonable Ineffectiveness of the Deeper Layers}, 
    author={Andrey Gromov and Kushal Tirumala and Hassan Shapourian and Paolo Glorioso and Daniel A. Roberts},
    year={2024},
    eprint={2403.17887},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@misc{song2024sleb,
    title={SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks}, 
    author={Jiwon Song and Kyungseok Oh and Taesu Kim and Hyungjun Kim and Yulhwa Kim and Jae-Joon Kim},
    year={2024},
    eprint={2402.09025},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@article{raecompressive2019,
    author = {Rae, Jack W and Potapenko, Anna and Jayakumar, Siddhant M and Hillier, Chloe and Lillicrap, Timothy P},
    title = {Compressive Transformers for Long-Range Sequence Modelling},
    journal = {arXiv preprint},
    url = {https://arxiv.org/abs/1911.05507},
    year = {2019},
}
```

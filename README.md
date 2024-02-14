# 🌋 Every Layer, Everywhere, All at Once: Segmenting Subsurface

<img src='assets/banner.png'>

The competition challenges participants to utilize Meta's Segment Anything Model (SAM) for new use cases beyond
traditional segmentation tasks in 3D seismic data analysis. The primary objective is to develop solutions capable of
identifying and mapping all layers within seismic data simultaneously. By doing so, participants aim to accelerate the
interpretation process, enabling quicker analysis of large datasets and fostering a deeper understanding of Earth's
structure and geological features. The provided dataset consists of approximately 9,000 pre-interpreted seismic volumes,
each accompanied by segment masks for model training. These volumes represent diverse geological settings and present
typical challenges of seismic data interpretation, including complex geology and data processing workflows. The holdout
data for evaluation mirrors the complexity of the training data, ensuring robust solutions capable of handling diverse
geologic features across different seismic volumes.

This project was made possible by our compute partners [2CRSI](https://2crsi.com/)
and [NVIDIA](https://www.nvidia.com/).

## 🏆 Challenge ranking

## 🛠️ Data processing

## 🏛️ Model architecture

<img src='assets/approach.png'>

## #️⃣ Command lines

### Launch a training

```bash
python src/models/<nom du model>/train_model.py <hyperparams args>
```

View project's runs on [WandB](https://wandb.ai/association-rosia/segmenting-subsurface/).

### Create a submission

```bash
python src/models/predict_model.py -n {model.ckpt}
```

## 🔬 References

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything.
arXiv preprint arXiv:2304.02643.

Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention mask transformer for
universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp.
1290-1299).

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for
semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34, 12077-12090.

## 📝 Citing

```
@misc{UrgellReberga:2023,
  Author = {Louis Reberga and Baptiste Urgell},
  Title = {Segmenting Subsurface},
  Year = {2024},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/association-rosia/segmenting-subsurface}}
}
```

## 🛡️ License

Project is distributed under [MIT License](https://github.com/association-rosia/segmenting-subsurface/blob/main/LICENSE)

## 👨🏻‍💻 Contributors

Louis
REBERGA <a href="https://twitter.com/rbrgAlou"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/louisreberga/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="louis.reberga@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a>

Baptiste
URGELL <a href="https://twitter.com/Baptiste2108"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/baptiste-urgell/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="baptiste.u@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a> 
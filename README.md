<img src="./gsa.png" width="500px"></img>

## Global Self-attention Network

An implementation of <a href="https://openreview.net/forum?id=KiFeuZu24k">Global Self-Attention Network</a>, which proposes an all-attention vision backbone that achieves better results than convolutions with less parameters and compute.

They use a previously discovered <a href="https://arxiv.org/abs/1812.01243">linear attention variant</a> with a small modification for further gains (no normalization of the queries), paired with relative positional attention, computed axially for efficiency.

The result is an extremely simple circuit composed of 7-8 einsums, 1 softmax, and normalization.

## Citations

```bibtex
@inproceedings{
    anonymous2021global,
    title={Global Self-Attention Networks},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=KiFeuZu24k},
    note={under review}
}
```

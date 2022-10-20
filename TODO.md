# TODO

1. Open an issue on torchmetrics for LPIPS:

```python
# RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your training graph has changed in this iteration, e.g., one parameter is used in first iteration, but then got unused in the second iteration. this is not compatible with static_graph set to True.                                                         Parameters which did not receive grad for rank 0: _forward_module.train_metrics.LearnedPerceptualImagePatchSimilarity.net.lin4.model.1.weight, _forward_module.train_metrics.LearnedPerceptualImagePatchSimilarity.net.lin3.model.1.weight, _forward_module.train_metrics.LearnedPerceptualImagePatchSimilarity.net.lin2.model.1.weight, _forward_module.train_metrics.LearnedPerceptualImagePatchSimilarity.net.lin1.model.1.weight, _forward_module.train_metrics.LearnedPerceptualImagePatchSimilarity.net.lin0.model.1.weight  
lpips =  LPIPS(net_type="alex", normalize=True).requires_grad_(False)
metrics = MetricCollection([MeanSquaredError(squared=True),lpips, SSIM(data_range=1.0), MS_SSIM(data_range=1.0), PSNR(data_range=1.0)])
```

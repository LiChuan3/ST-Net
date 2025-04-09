<div align="center">
  <h2><b> ST-Net: Dual-Branch Encoding with Seasonal-Trend Decomposition for Time Series Forecasting </b></h2>
</div>

## Introduction
 **ST-Net**, as a fully MLP-based architecture, taking full advantage of disentangled multiscale time series, is proposed to **achieve consistent SOTA performances in both long and short-term forecasting tasks with favorable run-time efficiency**.

🌟**Observation 1: History Extraction** 

Given that seasonal and trend components exhibit significantly different characteristics in time series, and different scales of the time series reflect different properties, with seasonal characteristics being more pronounced at a fine-grained micro-scale and trend characteristics being more pronounced at a coarse macro scale, it is, therefore, necessary to decouple seasonal and trend components at different scales.

<p align="center">
<img src="./figures/motivation1.png"  alt="" align=center />
</p>

🌟**Observation 2: Future Prediction** 

Different scales exhibit complementary predictive capabilities when integrating forecasts from different scales to obtain the final prediction results.

<p align="center">
<img src="./figures/motivation2.png"  alt="" align=center />
</p>

## Overall Architecture
TimeMixer as a fully MLP-based architecture with **Past-Decomposable-Mixing (PDM)** and **Future-Multipredictor-Mixing (FMM)** blocks to take full advantage of disentangled multiscale series in both past extraction and future prediction phases. 

<p align="center">
<img src="./figures/overall.png"  alt="" align=center />
</p>

### Past Decomposable Mixing 
we propose the **Past-Decomposable-Mixing (PDM)** block to mix the decomposed seasonal and trend components in multiple scales separately. 

<p align="center">
<img src="./figures/past_mixing1.png"  alt="" align=center />
</p>

Empowered by seasonal and trend mixing, PDM progressively aggregates the detailed seasonal information from fine to coarse and dive into the macroscopic trend information with prior knowledge from coarser scales, eventually achieving the multiscale mixing in past information extraction.

<p align="center">
<img src="./figures/past_mixing2.png"  alt="" align=center />
</p>

### Future Multipredictor Mixing 
Note that **Future Multipredictor Mixing (FMM)** is an ensemble of multiple predictors, where different predictors are based on past information from different scales, enabling FMM to integrate complementary forecasting capabilities of mixed multiscale series.

<p align="center">
<img src="./figures/future_mixing.png"  alt="" align=center />
</p>



## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download all datasets from [Google Driver](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download), [Baidu Driver](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) or [Kaggle Datasets](https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh
bash ./scripts/long_term_forecast/ECL_script/TimeMixer.sh
bash ./scripts/long_term_forecast/Traffic_script/TimeMixer.sh
bash ./scripts/long_term_forecast/Solar_script/TimeMixer.sh
bash ./scripts/long_term_forecast/Weather_script/TimeMixer.sh
bash ./scripts/short_term_forecast/M4/TimeMixer.sh
bash ./scripts/short_term_forecast/PEMS/TimeMixer.sh
```

## Main Results
We conduct extensive experiments to evaluate the performance and efficiency of TimeMixer, covering long-term and short-term forecasting, including 18 real-world benchmarks and 15 baselines.
**🏆 TimeMixer achieves consistent state-of-the-art performance in all benchmarks**, covering a large variety of series with different frequencies, variate numbers and real-world scenarios.

### Long-term Forecasting

To ensure model comparison fairness, experiments were performed with standardized parameters, aligning input lengths, batch sizes, and training epochs. Additionally, given that results in various studies often stem from hyperparameter optimization, we include outcomes from comprehensive parameter searches.

<p align="center">
<img src="./figures/long_results.png"  alt="" align=center />
</p>

### Short-term Forecasting: Multivariate data

<p align="center">
<img src="./figures/pems_results.png"  alt="" align=center />
</p>

###  Short-term Forecasting: Univariate data

<p align="center">
<img src="./figures/m4_results.png"  alt="" align=center />
</p>


## Model Abalations

To verify the effectiveness of each component of TimeMixer, we provide the detailed ablation study on every possible design in both Past-Decomposable-Mixing and Future-Multipredictor-Mixing blocks on all 18 experiment benchmarks （see our paper for full results 😊）.

<p align="center">
<img src="./figures/ablation.png"  alt="" align=center />
</p>

## Model Efficiency
We compare the running memory and time against the latest state-of-the-art models under the training phase, where TimeMixer consistently demonstrates favorable efficiency, in terms of both GPU memory and running time, for various series lengths (ranging from 192 to 3072), in addition to the consistent state-of-the-art performances for both long-term and short-term forecasting tasks.
**It is noteworthy that TimeMixer, as a deep model, demonstrates results close to those of full-linear models in terms of efficiency. This makes TimeMixer promising in a wide range of scenarios that require high model efficiency.**

<p align="center">
<img src="./figures/efficiency.png"  alt="" align=center />
</p>


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Autoformer (https://github.com/thuml/Autoformer)

# ML4DQM
Data preparation and analysis using ML to detect anomalies in HCAL.
The goal of the project is to:
   * create a tool to assist the shifters in inspecting the quality plots from DQM in real time and spot potential problems with the small possible delay
   * systematically study the possible net architectures and compare their performance on real data from HCAL (NN? CNN? AE? CAE?)
   * study different eras corresponding to different detector configurations (2016, 2017, 2018) and different dataset (ZeroBias, JetHT, ...)


### Data Preparation
The idea is to re-use as much as possible the existing DQM infrastructure. To start simple one long GOOD run from 2017 was processed from RAW. The processing spits out 1 file per LS containing the histograms from `RecHitTask` and `DigiTask` relative to that statistics.

Since the amount of BAD data wasn't large in 2016 and 2017, the idea is to process GOOD data only. BAD data can be created manually starting from GOOD data and mimiking the most common issues (e.g. killing manually some tower/module in the occupancy plots, raise the noise in some regions, etc).

To easily handle the data, rootfiles are created and stored on EOS. They contain all histograms created by the aforementioned DQM tasks. A subset of histograms is then converted into numpy arrays and saved in the data folder of this github repo.

To reprocess the data selecting a different set of histos one could:
   * edit `convertHistToArr.py`
   * `source /cvmfs/sft.cern.ch/lcg/views/LCG_88/x86_64-slc6-gcc49-opt/setup.csh`
   * `python convertHistToArr.py`


### Data Location
histograms: `/eos/cms/store/group/dpg_hcal/comm_hcal/ML4DQM/process_2017/submit_20180304_142701/` (1000 LS from run 306138)
numpy: `data/HCAL_digi+rechit_occ.hdf5`


### Models
In general two approaches are possible:
   * Supervised model: train a CNN on GOOD and BAD images
   * Semi-supervised model: use GOOD data only to train the algorithms and reject the reconstructed data with large error function


### How to use this repo
   * For data processing this repo can be cloned in CMSSW
   * For testing a model and for plotting, jupiter notebooks can be used. They allow to produce easily plots and store them directly in github. One option is to use swan.cern.ch. One example is: https://github.com/HCALDQM/ML4DQM/blob/master/notebooks/plotMaps.ipynb
   * For training, which is CPU intensive, notebooks are not a good option. Eventually it would be nice to move to GPU. 
   


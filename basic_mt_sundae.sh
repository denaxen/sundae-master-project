cd sundae
pip install -r requirements.txt
CUDA_LAUNCH_BLOCKING=1 python main.py data=wmt14 model=sundae_diffusion_mt 'model.trainer.callbacks=[{_target_: callbacks.TranslationSamplingCallback, sample_frequency: 1000, nb_samples: 1}]'

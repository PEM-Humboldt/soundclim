# MAKE AUDIO DATABSE FOR MANUAL ANNOTATION
# Create an audio file with segments for easly annotate and create an object for machine learning on the style of scykit learn.
# input: rois with fname and labels (optional)
# 1. MAAD Python - compile_dataset.py
# 2. Make dataset object with selected observatiobs - format_trainds
# 3. Make audio db for manual annotation - mix_audio_trainds.R
# 4. Inspect audio manually


library(seewave)
library(tuneR)
# Set variables
load_name =  '~/Dropbox/PostDoc/iavh/visita_UAM/detector_shayii/data_training/traindb_RDATA'
save_name = '~/Dropbox/PostDoc/iavh/visita_UAM/detector_shayii/data_training/traindb_shayii.wav'

# load db and assign variables
load(load_name)
audio_seg=train_data$audio
seg=train_data$segments

wl=2  # window of each sample in seconds

# assign names
fs=audio_seg[[1]]@samp.rate
bit=audio_seg[[1]]@bit
pcm=audio_seg[[1]]@pcm

# concatenate audio
sx=lapply(audio_seg,function(x){x@left})
sx=do.call(c,sx)
sx=Wave(sx,samp.rate = fs, bit = bit, pcm = pcm)
sx=normalize(sx,unit = '16',level = 0.7)

# create annotations
onset=(0:(nrow(seg)-1)*wl)+seg$onset
offset=(0:(nrow(seg)-1)*wl)+seg$offset
seg=data.frame(onset,offset)
seg$label=1:nrow(seg)

# write on disk
write.table(seg,file= paste(save_name,'.txt',sep=''),sep = '\t',col.names = F,row.names = F)
writeWave(sx,filename = paste(save_name,'.wav',sep=''))

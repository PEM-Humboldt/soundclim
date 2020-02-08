## Make_dataset_object_with_selected_observations
# Through this script, all information is formated into a single R dataset
# Pipeline:
# 1. MAAD Python - compile_dataset.py
# 2. Format audio dataset for manual annotation - format_trainds.R
# 3. Inspect audio manually
# 4. Match annotations woth features - export_features_and_mannot_to_csv.R
# 5. Train calssifier tune_clf_simple.py

## NOTA: SE DEBE MODIFICAR LA ULTIMA PARTE DONDE SE GUARDA TODO EN LA LISTA
# SE ESTÁ GUARDANDO SOLO SHAPE FEATURES


# --- begin header -- ##
# load libraries
library(tuneR)
library(seewave)
# variables
# set options
sp_name = 'scinhayii'; flims=c(900,5000) # in Hz
wl=2 # window for sample in seconds for audio
audio_db_path='~/Dropbox/PostDoc/Soundclim/audio_sites/BETANIA/train/'
path_xdata_traindb = '~/Dropbox/PostDoc/iavh/visita_UAM/detector_shayii/data_training/df_stratsample.csv'
path_save_train_db = '~/Dropbox/PostDoc/iavh/visita_UAM/detector_shayii/data_training/traindb_RDATA'
# --- end header -- ##

## LOAD XDATA TRAINDB
dbsel = read.table(path_xdata_traindb, header = TRUE, sep=',')
head(dbsel)


# Concatenate each segment into a unique file  
# modify samples that are longer than wl
audiolist=list()
db=cbind(dbsel['fname'],dbsel['min_t'],dbsel['max_t'])
db$width = db$max_t-db$min_t
db[which(db$width>1),'width']<-1
for(i in 1:nrow(db)){
  roi=db[i,]
  fname_wav=paste(audio_db_path,roi$fname,sep='')
  tlims=c(roi$min_t,roi$max_t)
  # define tlimits with window length
  len=tlims[2]-tlims[1]
  tlims=c((tlims[1]+len/2)-wl/2,(tlims[1]+len/2)+wl/2)
  # read wave and normalize
  aux=readWave(fname_wav,from = tlims[1],to = tlims[2],units = 'seconds')
  metadata_wav=readWave(fname_wav, header=TRUE)
  aux=fir(aux,from=flims[1],to=flims[2],bandpass=T,output='Wave')
  aux=normalize(aux,unit = '16',level = 0.7)
  # if time limits are outside the recording, add silence
  rec_length=metadata_wav$samples/metadata_wav$sample.rate
  if(tlims[2]>rec_length){
    sil_length=tlims[2]-rec_length
    aux=addsilw(aux,at='end',d=sil_length,output = 'Wave')
  }else if(tlims[1]<0){
    sil_length=abs(tlims[1])
    aux=cutw(aux,from = sil_length, to = wl,output = 'Wave') # tuneR adds signal when reading
    aux=addsilw(aux,at='start',d=sil_length,output = 'Wave')
  }else{
    aux=aux #unchanged
  }
  # save waves
  audiolist[[i]]=aux
}

# write segments for manual annotations
onset=(wl/2)-(db$width/2)
offset=(wl/2)+(db$width/2)
seg=data.frame(onset,offset)
seg$label=NA

## assign to object and save
idx_features = c(grep('shp',colnames(dbsel)),grep('frequency',colnames(dbsel)))
train_data=list()
train_data[['roi_info']]=dbsel[c('fname','min_t','max_t','min_f','max_f')]
train_data[['shape_features']]=dbsel[idx_features] 
train_data[['label']]=seg$label
train_data[['audio']]=audiolist
train_data[['segments']]=seg[c('onset','offset')]
train_data[['maad_label']]=dbsel$maad_label
save(train_data,file = path_save_train_db)


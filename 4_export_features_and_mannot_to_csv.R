# EXPORT DB TO CSV
# Mix RData DB and labels to a standard csv file. 
# Makes the db compatible with pyhton

#---------- header ---------
fname_db='~/Dropbox/PostDoc/Soundclim/compile_dataset/scinhayii_campobelo_alt/traindb_600.RData'
fname_mannot='~/Dropbox/PostDoc/Soundclim/compile_dataset/scinhayii_campobelo_alt/traindb_scinhayii_600_mannot.txt'
fname_save='~/Dropbox/PostDoc/Soundclim/compile_dataset/scinhayii_campobelo_alt/scinhayii_traindb_features_mannot.csv'
# --------------------------

# load data
load(fname_db)
gt=read.table(fname_mannot,header = F)

# format
colnames(gt)<-c('onset','offset','label')
lab_wname = gt$label

# check typo errors
table(lab_wname)

aux = strsplit(as.character(lab_wname),'_')
lab_gt = as.factor(unlist(lapply(aux, function(l) l[[1]])))
idx_xdata = as.numeric(row.names(train_data$roi_info))

db_mannot = cbind(train_data$shape_features, 
                  train_data$roi_info,
                  data.frame('lab_gt'=lab_gt, 
                             'lab_wname'=lab_wname,
                             'idx_xdata' = idx_xdata,
                             'obs'=1:nrow(train_data$shape_features)))

# save as csv file
write.table(db_mannot, fname_save, sep=',', col.names = T, row.names = F)



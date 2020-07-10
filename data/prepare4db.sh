# less train_raw.tsv| awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}'> train.tsv
less dd_all_train.tsv| awk -F '\t' '{print "0.0 "$1"\t1.0 "$2"\t2.0 "$3"\t3.0 "$4}'> train.tsv
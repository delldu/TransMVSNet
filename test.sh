#!/usr/bin/env bash
TESTPATH="data/dtu_test" 						# path to dataset dtu_test
TESTLIST="data/dtu_test/list.txt"
CKPT_FILE="checkpoints/model_dtu.ckpt"			   # path to checkpoint file, you need to use the model_dtu.ckpt for testing
FUSIBLE_PATH="gipuma/fusibile/build/fusibile" 								 	# path to fusible of gipuma
OUTDIR="outputs/dtu_testing" 						  # path to output
BATCH_SIZE=3

if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


python test.py \
--dataset=general_eval \
--batch_size=$BATCH_SIZE \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--fusibile_exe_path=$FUSIBLE_PATH
#--filter_method="normal"


# gipuma/fusibile/build/fusibile outputs/dtu_testing/scan24


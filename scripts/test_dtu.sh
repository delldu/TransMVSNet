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
--numdepth=192 \
--ndepths="48,32,8" \
--depth_inter_r="4.0,1.0,0.5" \
--interval_scale=1.06 \
--filter_method="gipuma" \
--fusibile_exe_path=$FUSIBLE_PATH
#--filter_method="normal"


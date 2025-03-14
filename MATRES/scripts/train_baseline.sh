CUDA_VISIBLE_DEVICES=6 python train_matres.py --use_tense 0 \
	--epochs 10 \
	--scheduler linear \
	--train_json_path data/train_features_matres.json \
	--valid_json_path data/valid_features_matres.json \
	--test_json_path data/test_features_matres.json

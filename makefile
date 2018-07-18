version = 0.1
data_dir_host = /mnt/nas/datasets/ICDAR
data_dir_docker = /data
checkpoint_dir_host = /mnt/nas/tf_experiments/ocr/east/training/3
checkpoint_dir_docker = /training
checkpoint_path_docker = $(checkpoint_dir_docker)/east

frozen_model_path = /mnt/nas/tf_experiments/ocr/east/frozen/frozen_east2.pb

docker:
	docker build -t seb/train_east:$(version) .
train:
	nvidia-docker run -it --rm  \
	-v $(data_dir_host):$(data_dir_docker) \
	-v $(checkpoint_dir_host):$(checkpoint_dir_docker) \
	seb/train_east:$(version) \
	bash -c "python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=8 \
	--checkpoint_path=$(checkpoint_path_docker) --text_scale=512 --training_data_path=$(data_dir_docker)/merge \
	--geometry=RBOX --learning_rate=0.0001 --num_readers=24  --restore=false"
demo_server:
	nvidia-docker run -it --rm -v $(frozen_model_path):/root/frozen_east.pb \
	-p 8769:8769 \
	seb/train_east:$(version) bash -c "python run_demo_server.py /root/frozen_east.pb"

#prod_server:
#	nvidia-docker run -it --rm -v $(frozen_model_path):/root/frozen_east.pb \
#	-p 8769:8769 \
#	seb/train_east:$(version) bash -c "python run_demo_server.py /root/frozen_east.pb"
#gunicorn -w 3 run_demo_server:app -b 0.0.0.0:8769 -t 120 --error-logfile server_log/error.log \
#	--access-logfile server_log/access.log


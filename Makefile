#################### PACKAGE ACTIONS ###################
install:
	@pip install -e .

reinstall:
	@pip uninstall -y facetally || :
	@pip install -e .

update_raw_data_from_GCP:
	python -c 'from face_tally.ml_logic.data import update_local_raw_data_from_GCP; update_local_raw_data_from_GCP()'

test_preproc:
	@python face_tally/visulization/visualisation.py

create_folders:
	@mkdir ~/.lewagon/project_data
	@mkdir ~/.lewagon/project_data/image_data
	@echo Successfully created folders

copy_from_old_structure:
	@cp raw_data/bbox_train.csv ~/.lewagon/project_data/bbox_train.csv
	@cp raw_data/train.csv ~/.lewagon/project_data/train.csv
	@cp -r raw_data/image_data ~/.lewagon/project_data/

test_training:
	@python face_tally/visulization/visualisation.py
